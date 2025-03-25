# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy only requirements.txt first for efficient caching
COPY requirements.txt /app/requirements.txt

# Step 4: Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*  
    
    # Clean up apt cache to reduce image size

# Step 5: Copy the current directory contents into the container at /app
COPY . /app

# Step 6: Expose the port your app will run on (default Flask runs on 5000)
EXPOSE 5000

# Step 7: Run the application using Gunicorn, binding to 0.0.0.0 to allow external connections
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
