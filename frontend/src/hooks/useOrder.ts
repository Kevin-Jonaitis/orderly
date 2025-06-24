import { useState, useEffect, useRef, useCallback } from 'react';
import { Order } from '../types/order';

// Simple UUID generator for frontend keys
function generateUUID() {
  return 'frontend-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

export function useOrder() {
  const [order, setOrder] = useState<Order>({ items: [], total: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const websocketRef = useRef<WebSocket | null>(null);

  // Process order data to add frontend IDs
  const processOrderData = useCallback((orderData: any) => {
    const itemsWithFrontendIds = orderData.items.map((item: any) => ({
      ...item,
      frontendId: item.frontendId || generateUUID()
    }));
    return { ...orderData, items: itemsWithFrontendIds };
  }, []);

  // Fetch current order via REST API (fallback)
  const fetchOrder = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/order');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const orderData = await response.json();
      const processedOrder = processOrderData(orderData);
      setOrder(processedOrder);
    } catch (error) {
      console.error('Error fetching order:', error);
      setOrder({ items: [], total: 0 });
    } finally {
      setIsLoading(false);
    }
  };

  // Connect to order WebSocket
  const connectWebSocket = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    console.log('Connecting to Order WebSocket...');
    const ws = new WebSocket('ws://localhost:8000/ws/order');
    
    ws.onopen = () => {
      console.log('Order WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const orderData = JSON.parse(event.data);
        const processedOrder = processOrderData(orderData);
        setOrder(processedOrder);
        setIsLoading(false);
      } catch (error) {
        console.error('Error parsing order WebSocket message:', error);
      }
    };

    ws.onclose = (event) => {
      console.log('Order WebSocket disconnected', event.code, event.reason);
      setIsConnected(false);
      
      // Try to reconnect after a delay unless it was a clean disconnect
      if (event.code !== 1000) {
        setTimeout(() => {
          connectWebSocket();
        }, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error('Order WebSocket error:', error);
      setIsConnected(false);
      
      // Fallback to REST API
      fetchOrder();
    };

    websocketRef.current = ws;
  }, [processOrderData]);

  // Clear order
  const clearOrder = async () => {
    try {
      await fetch('http://localhost:8000/api/order/clear', {
        method: 'POST',
      });
      // Don't manually update order here - let WebSocket update handle it
    } catch (error) {
      console.error('Error clearing order:', error);
    }
  };

  // Connect to WebSocket on mount
  useEffect(() => {
    setIsLoading(true);
    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    };
  }, [connectWebSocket]);

  return {
    order,
    isLoading,
    isConnected,
    clearOrder,
    refetch: fetchOrder,
  };
}