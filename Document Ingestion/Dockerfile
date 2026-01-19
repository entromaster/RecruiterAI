#docker base image
FROM mcr.microsoft.com/devcontainers/python:3.10

# Set the working directory in the container
WORKDIR /app


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git libsm6 libxext6 libxrender-dev libgl1-mesa-dev libglib2.0-0 libnss3 libnss3-dev libcairo2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy the current directory contents into the container
COPY . /app

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 443

# Define the command to run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "443"]