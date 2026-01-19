from typing import Dict, List, Any, Callable, Awaitable, Optional, Tuple
import asyncio
from asyncio import Queue, Event
import logging

from agentic_rag.src.utils import (
    get_traceback
)
from agentic_rag.src import (
    WORKER_AND_BATCH_SIZE
)
from agentic_rag.processor.batch_processor import (
    process_base_endo_batch,
    process_declaration_batch
)
from agentic_rag.service_bus import receive_messages

ProcessorFunction = Callable[[List[Any]], Awaitable[None]]
stop_event = Event()
receive_messages_started = False

class PriorityTimer:
    def __init__(self, timeout: float, callback: Callable[[], Awaitable[None]]):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())
        self._cancelled = False

    async def _job(self):
        try:
            await asyncio.sleep(self._timeout)
            await self._callback()
        except asyncio.CancelledError:
            self._cancelled = True
            raise

    def cancel(self):
        self._task.cancel()

    @property
    def was_cancelled(self):
        return self._cancelled
    
class QueueState:
    def __init__(self, queue: Queue, priority: int, name: str, 
                 idle_timeout: Optional[float] = None, 
                 idle_callback: Optional[Callable[[], Awaitable[None]]] = None,
                 logger: Optional[logging.Logger] = None):
        self.queue = queue
        self.priority = priority
        self.name = name
        self.idle_timeout = idle_timeout
        self.idle_callback = idle_callback
        self.timer = None
        self.logger = logger or logging.getLogger('PriorityQueueSystem')
        self.idle_condition_met = Event()
        if not idle_timeout or not idle_callback:
            self.idle_condition_met.set()
    
    def is_empty(self) -> bool:
        return self.queue.empty()
    
    def is_processable(self) -> bool:
        return not self.is_empty() and self.idle_condition_met.is_set()
    
    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if self.idle_timeout and self.idle_callback:
            self.idle_condition_met.clear()
            self.timer = PriorityTimer(self.idle_timeout, self.idle_callback)
            self.logger.info(f"Started idle timer for {self.name} queue: {self.idle_timeout}s")

class PriorityQueueManager:
    def __init__(self, logger=None):
        self.queue_states: Dict[str, QueueState] = {}
        self.logger = logger or logging.getLogger('PriorityQueueManager')
    
    @classmethod
    def from_config(cls, queue_dict: Dict[str, Any], logger=None):
        manager = cls(logger)
        return manager
    
    def configure_queues(self, queue_dict: Dict[str, Any]) -> None:
        for queue_name, config in queue_dict.items():
            queue = globals().get(queue_name)
            if queue is None:
                self.logger.warning(f"Queue '{queue_name}' not found in globals, skipping")
                continue
            if isinstance(config, int):
                self.register_queue(
                    name=queue_name,
                    queue=queue,
                    priority=config
                )
            elif isinstance(config, dict):
                priority = config.get("priority", 0)
                idle_timeout = config.get("idle_timeout")
                min_priority_level = config.get("min_priority_level")
                self.register_queue(
                    name=queue_name,
                    queue=queue,
                    priority=priority,
                    idle_timeout=idle_timeout,
                    min_priority_level=min_priority_level
                )
            else:
                self.logger.warning(f"Invalid configuration for queue '{queue_name}', skipping")
        
    def register_queue(self, name: str, queue: Queue, priority: int = 0, 
                      idle_timeout: Optional[float] = None,
                      min_priority_level: Optional[int] = None) -> None:
        effective_min_priority = min_priority_level if min_priority_level is not None else priority + 1
        async def generic_idle_callback():
            try:
                are_higher_queues_empty = self.are_all_queues_empty_above_priority(effective_min_priority)
                if are_higher_queues_empty:
                    self.logger.info(
                        f"Idle condition met for {name}: All queues with priority >= {effective_min_priority} are empty"
                    )
                    self.queue_states[name].idle_condition_met.set()
                else:
                    self.logger.info(
                        f"Idle condition NOT met for {name}: Some queues with priority >= {effective_min_priority} still have items"
                    )
                    self.queue_states[name].idle_condition_met.clear()
            except Exception as e:
                self.logger.error(f"Error in idle callback for {name}: {str(e)}")
                self.queue_states[name].idle_condition_met.set()
        self.queue_states[name] = QueueState(
            queue=queue,
            priority=priority,
            name=name,
            idle_timeout=idle_timeout,
            idle_callback=generic_idle_callback if idle_timeout else None,
            logger=self.logger
        )
        self.logger.info(
            f"Registered queue '{name}' with priority {priority}" +
            (f", idle timeout {idle_timeout}s, checking priority >= {effective_min_priority}" 
             if idle_timeout else "")
        )
        
    def are_all_queues_empty_above_priority(self, min_priority: int) -> bool:
        for name, state in self.queue_states.items():
            if state.priority >= min_priority and not state.is_empty():
                self.logger.debug(f"Queue '{name}' with priority {state.priority} is not empty")
                return False
        return True
    
    def update_priority(self, name: str, priority: int) -> bool:
        if name in self.queue_states:
            self.queue_states[name].priority = priority
            self.logger.info(f"Updated queue '{name}' to priority {priority}")
            return True
        return False
    
    def get_queue_by_name(self, name: str) -> Optional[Queue]:
        if name in self.queue_states:
            return self.queue_states[name].queue
        return None
    
    def get_next_processable_queue(self) -> Optional[Tuple[str, Queue]]:
        sorted_states = sorted(
            self.queue_states.items(), 
            key=lambda x: x[1].priority
        )
        for name, state in sorted_states:
            if state.is_processable():
                return name, state.queue
        return None
    
    def check_idle_conditions(self) -> None:
        sorted_states = sorted(
            self.queue_states.items(), 
            key=lambda x: x[1].priority
        )
        for name, state in sorted_states:
            if state.idle_timeout and not state.idle_condition_met.is_set() and state.timer is None:
                if self.are_all_queues_empty_above_priority(state.priority + 1):
                    state.reset_timer()
                    self.logger.info(
                        f"Starting idle timer for {name}: All higher priority queues are empty"
                    )
        
    def reset_timers_for_non_empty_queues(self) -> None:
        for name, state in self.queue_states.items():
            if not state.is_empty() and state.idle_timeout:
                if state.timer:
                    state.timer.cancel()
                    state.timer = None
                state.idle_condition_met.clear()
                self.logger.debug(f"Reset idle condition for {name} queue (not empty)")
                
    def get_queue_status_summary(self) -> str:
        summary = ["Queue Status Summary:"]
        sorted_states = sorted(
            self.queue_states.items(), 
            key=lambda x: x[1].priority
        )
        for name, state in sorted_states:
            status = "READY" if state.is_processable() else "WAITING"
            if state.is_empty():
                count = "empty"
            else:
                count = "not empty"  # For asyncio.Queue we can't get size without emptying
            if state.idle_timeout:
                idle_status = "met" if state.idle_condition_met.is_set() else "not met"
                timer_status = "active" if state.timer else "inactive"
                summary.append(
                    f"  {name} (priority {state.priority}): {status}, {count}, " +
                    f"idle condition {idle_status}, timer {timer_status}"
                )
            else:
                summary.append(
                    f"  {name} (priority {state.priority}): {status}, {count}"
                )
        return "\n".join(summary)
    
async def process_queue_batch(queue_name: str, queue: Queue, batch_size: int, 
                             processor_func, logger) -> bool:
    if queue.empty():
        logger.debug(f"Queue {queue_name} is empty, nothing to process")
        return False
    batch = []
    try:
        for _ in range(batch_size):
            if queue.empty():
                break
            batch.append(queue.get_nowait())
    except asyncio.QueueEmpty:
        pass
    if not batch:
        logger.debug(f"No items collected from {queue_name} queue")
        return False
    try:
        logger.info(f"Processing batch of {len(batch)} items from {queue_name} queue")
        await processor_func(batch)
        logger.info(f"Successfully processed {len(batch)} items from {queue_name} queue")
        return True
    except Exception as e:
        err_msg = get_traceback(e, f"Error processing {queue_name} batch:")
        logger.error(err_msg)
        logger.info(f"Returning {len(batch)} items back to {queue_name} queue after error")
        for item in batch:
            await queue.put(item)
        return False                   
    
async def process_batches(
    queue_manager: PriorityQueueManager,
    processor_functions: Dict[str, ProcessorFunction],
    batch_size: int,
    stop_event: asyncio.Event,
    properties: Dict[str, Any],
    app_insight_logger
) -> None:
    app_insight_logger.info("Starting continuous process_batches with priority queue system", extra=properties)
    iterations = 0
    try:
        last_queue_processed = None
        while not stop_event.is_set():
            if stop_event.is_set():
                app_insight_logger.info("Stop event detected, process_batches will exit", extra=properties)
                break
            iterations += 1
            if iterations % 100 == 0:
                app_insight_logger.info(f"Process_batches still running - iteration {iterations}", extra=properties)
                if stop_event.is_set():
                    app_insight_logger.info("Stop event detected during iteration check", extra=properties)
                    break
            try:
                queue_manager.check_idle_conditions()
                if iterations % 200 == 0:
                    status_summary = queue_manager.get_queue_status_summary()
                    app_insight_logger.info(status_summary, extra=properties)
                queue_info = queue_manager.get_next_processable_queue()
                if queue_info:
                    queue_name, queue = queue_info
                    app_insight_logger.info(f"Selected {queue_name} for processing", extra=properties)
                    if stop_event.is_set():
                        app_insight_logger.info("Stop event detected before processing queue", extra=properties)
                        break
                    if queue_name in processor_functions:
                        processed = await process_queue_batch(
                            queue_name=queue_name,
                            queue=queue,
                            batch_size=batch_size,
                            processor_func=processor_functions[queue_name],
                            logger=app_insight_logger
                        )
                        if processed:
                            last_queue_processed = queue_name
                            if queue.empty():
                                app_insight_logger.info(f"{queue_name} queue is now empty, checking idle conditions", 
                                                     extra=properties)
                                queue_manager.check_idle_conditions()
                            queue_manager.reset_timers_for_non_empty_queues()
                    else:
                        app_insight_logger.error(f"No processor function found for queue {queue_name}", 
                                              extra=properties)
                else:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                        if stop_event.is_set():
                            app_insight_logger.info("Stop event detected during wait", extra=properties)
                            break
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                    if stop_event.is_set():
                        app_insight_logger.info("Stop event detected during pause", extra=properties)
                        break
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                app_insight_logger.info("Process batches received cancellation", extra=properties)
                break
            except Exception as e:
                err_msg = get_traceback(e, "Error in process_batches loop:")
                app_insight_logger.error(err_msg, extra=properties)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                    if stop_event.is_set():
                        app_insight_logger.info("Stop event detected during error recovery", extra=properties)
                        break
                except asyncio.TimeoutError:
                    pass
        app_insight_logger.info("Exiting process_batches loop", extra=properties)
    except asyncio.CancelledError:
        app_insight_logger.info("Process batches task cancelled", extra=properties)
    except Exception as e:
        err_msg = get_traceback(e, "Fatal error in process_batches:")
        app_insight_logger.error(err_msg, extra=properties)
    finally:
        app_insight_logger.info(f"Closing process_batches after {iterations} iterations", extra=properties)


async def main(app_insight_logger, properties):
    queue_manager = PriorityQueueManager(logger=app_insight_logger)
    queue_config = {
        "base_endo_queue": 0,
        "declaration_queue": {
            "priority": -1,  # Lowest priority
            "idle_timeout": 40.0,  # 40 seconds
            "min_priority_level": 0  # Check all queues with priority 0 or higher
        }
    }
    queue_manager.configure_queues(queue_config)
    processor_functions = {
        "base_endo_queue": process_base_endo_batch,
        "declaration_queue": process_declaration_batch,
    }
    process_task = asyncio.create_task(
        process_batches(
            queue_manager=queue_manager,
            processor_functions=processor_functions,
            batch_size=WORKER_AND_BATCH_SIZE,
            stop_event=stop_event,
            properties=properties
        )
    )
    async def log_running_tasks():
        while not stop_event.is_set():
            app_insight_logger.info("All background tasks still running", extra=properties)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass
    monitoring_task = asyncio.create_task(log_running_tasks())
    receive_task = None
    if not receive_messages_started:
        receive_task = asyncio.create_task(receive_messages(queue_manager))
        receive_messages_started = True
        app_insight_logger.info("Starting the receive_messages().", extra=properties)
    else:
        app_insight_logger.info("receive_messages() was already started.", extra=properties)
    tasks = []
    if receive_task:
        tasks.append(receive_task)
    tasks.extend([
        process_task, 
        monitoring_task, 
    ])
    try:
        app_insight_logger.info("Starting continuous processing of all queues", extra=properties)
        await stop_event.wait()
        app_insight_logger.info("Stop event detected in main(), cancelling all tasks", extra=properties)
    except asyncio.CancelledError:
        app_insight_logger.info("Main task was cancelled", extra=properties)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        app_insight_logger.info("Waiting for all tasks to complete...", extra=properties)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            app_insight_logger.info("All tasks completed successfully", extra=properties)
        except asyncio.TimeoutError:
            app_insight_logger.info("Some tasks did not complete in time", extra=properties)
        except Exception as e:
            app_insight_logger.error(f"Error during task cleanup: {str(e)}", extra=properties)
        app_insight_logger.info("Main function completed, all tasks have been cancelled", extra=properties)