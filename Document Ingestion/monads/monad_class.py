from functools import wraps


class Monad:
    def __init__(self, value, is_success, exception=None):
        self.value = value
        self.is_success = is_success
        self.exception = exception  # Store the exception if present

    @staticmethod
    def Ok(value):
        return Monad(value, True)

    @staticmethod
    def Err(error):
        if isinstance(error, Exception):
            # Store the actual exception
            return Monad(error, False, exception=error)
        # Handle non-exception errors as a message
        return Monad(str(error), False)

    def map(self, func):
        if self.is_success:
            try:
                return Monad.Ok(func(self.value))
            except Exception as e:
                return Monad.Err(e)  # Pass the actual exception
        return self

    def unwrap(self):
        if self.is_success:
            return self.value
        if self.exception:
            raise self.exception  # Raise the original exception
        raise ValueError(f"Error: {self.value}")


def monad_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Call the function
            result = func(*args, **kwargs)
            # Wrap the result in Monad.Ok
            return Monad.Ok(result)
        except Exception as e:
            # Wrap the original exception in Monad.Err
            return Monad.Err(e)  # Pass the actual exception

    return wrapper


def async_monad_wrapper(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Call the async function and await the result
            result = await func(*args, **kwargs)
            return Monad.Ok(result)
        except Exception as e:
            # Wrap the original exception in Monad.Err
            return Monad.Err(e)  # Pass the actual exception

    return wrapper
