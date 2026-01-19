# import asyncio
# import os
# from threading import Thread
# import signal
# from flask import Flask
# import concurrent
# import time
# from agentic_rag.src import (
#     APP_NAME
# )
# from agentic_rag import (
#     azure_log,
#     main
# )
# from agentic_rag.src import (
#     ENV_NAME,
#     ENV_NAMES,
#     WEBSITE_PORT,
# )

# properties = properties = {'custom_dimensions': {'ApplicationName': APP_NAME}}
# app_insight_logger = azure_log(APP_NAME)
# background_loop = None
# main_pid = os.getpid()

# app = Flask(__name__)

# def start_background_loop(loop):
#     asyncio.set_event_loop(loop)
#     loop.run_forever()

# def run_main():
#     """Application entry point with improved shutdown handling."""
#     global background_loop, stop_event
#     app_insight_logger.info(f"Inside run_main() function in process id: {os.getpid()}", extra=properties)
#     background_loop = asyncio.new_event_loop()
#     t = Thread(target=start_background_loop, args=(background_loop,), daemon=True)
#     t.start()
#     main_future = None
#     if os.getpid() == main_pid:
#         main_future = asyncio.run_coroutine_threadsafe(main(app_insight_logger, properties), background_loop)
#     else:
#         app_insight_logger.info("Not scheduling main() since this is not the main process", extra=properties)
#     def signal_handler(sig, frame):
#         nonlocal main_future
#         app_insight_logger.info(f"Received signal {sig}, initiating shutdown", extra=properties)
#         background_loop.call_soon_threadsafe(stop_event.set)
#         if sig == signal.SIGINT and main_future and not main_future.done():
#             app_insight_logger.warning("Second interrupt received. Forcing exit.", extra=properties)
#             import sys
#             sys.exit(1)
#     signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
#     signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

#     try:
#         if ENV_NAME in ENV_NAMES:
#             app.run(
#                 debug=False,
#                 use_reloader=False,
#                 threaded=False,
#                 processes=1,
#                 port=WEBSITE_PORT,
#                 host="0.0.0.0",
#             )
#         else:
#             app.run(
#                 debug=True,
#                 use_reloader=False,
#                 threaded=False,
#                 processes=1,
#                 port=WEBSITE_PORT,
#             )
#     except KeyboardInterrupt:
#         app_insight_logger.info("Received KeyboardInterrupt in main thread", extra=properties)
#         background_loop.call_soon_threadsafe(stop_event.set)
#     finally:
#         app.logger.info("Starting shutdown sequence")
#         background_loop.call_soon_threadsafe(stop_event.set)
#         time.sleep(1)
#         if os.getpid() == main_pid and main_future:
#             try:
#                 main_future.result(timeout=3)
#                 app.logger.info("Main task completed successfully")
#             except concurrent.futures.TimeoutError:
#                 app.logger.warning("Main task did not complete in time, forcing shutdown")
#                 main_future.cancel()
#             except Exception as e:
#                 app.logger.error(f"Error in main task: {str(e)}")
#         try:
#             app.logger.info("Stopping background event loop")
#             background_loop.call_soon_threadsafe(lambda: [
#                 background_loop.stop(),
#                 app.logger.info("Event loop stop requested")
#             ])
#             time.sleep(0.5)
#             if not background_loop.is_closed():
#                 app.logger.warning("Event loop still running, forcing close")
#                 for task in asyncio.all_tasks(background_loop):
#                     background_loop.call_soon_threadsafe(task.cancel)
#                 background_loop.call_soon_threadsafe(background_loop.stop)
#                 time.sleep(0.5)
#         except Exception as e:
#             app.logger.error(f"Error stopping event loop: {str(e)}")
#         app.logger.info("Waiting for background thread to complete")
#         t.join(timeout=2)
#         if t.is_alive():
#             app.logger.warning("Background thread did not complete in time, proceeding with shutdown anyway")
#         else:
#             app.logger.info("Background thread completed successfully")
#         app.logger.info("Shutdown completed")

# if __name__ == "__main__":
#     app_insight_logger.info("Starting the application...", extra=properties)
#     run_main()