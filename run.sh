#!/bin/bash

# AI Order Taker - Start Script
# Runs both backend and frontend in development mode

echo "🚀 Starting AI Order Taker (Backend + Frontend)"
echo "🌐 Backend will be available at: http://localhost:8000"
echo "📖 API docs at: http://localhost:8000/docs"
echo "⚛️  Frontend will be available at: http://localhost:5173"
echo "⏹️  Press Ctrl+C to stop both servers"
echo

# Function to kill processes on exit
cleanup() {
    echo
    echo "👋 Stopping servers..."
    
    # Kill backend process
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        wait $BACKEND_PID 2>/dev/null
    fi
    
    # Kill frontend process
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        wait $FRONTEND_PID 2>/dev/null
    fi
    
    # Kill any remaining processes on these ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found! Please run:"
    echo "  python setup_env.py"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found! Please run:"
    echo "  cd frontend && npm install"
    exit 1
fi

# Set up CUDA environment for backend
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start backend in background
echo "🐍 Starting backend..."
source venv/bin/activate
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Give backend time to start
sleep 3

# Start frontend in background
echo "⚛️  Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "✅ Both servers started!"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo
echo "📍 URLs:"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait