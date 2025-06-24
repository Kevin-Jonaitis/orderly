import { useState, useEffect } from 'react';
import { Order } from '../types/order';

// Simple UUID generator for frontend keys
function generateUUID() {
  return 'frontend-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

export function useOrder() {
  const [order, setOrder] = useState<Order>({ items: [], total: 0 });
  const [isLoading, setIsLoading] = useState(false);

  // Fetch current order
  const fetchOrder = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:8000/api/order');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const orderData = await response.json();
      // Add frontend IDs to each item for React keys
      const itemsWithFrontendIds = orderData.items.map((item: any) => ({
        ...item,
        frontendId: item.frontendId || generateUUID()
      }));
      setOrder({ ...orderData, items: itemsWithFrontendIds });
    } catch (error) {
      console.error('Error fetching order:', error);
      // Set empty order on error so component can still render
      setOrder({ items: [], total: 0 });
    } finally {
      setIsLoading(false);
    }
  };

  // Clear order
  const clearOrder = async () => {
    try {
      await fetch('http://localhost:8000/api/order/clear', {
        method: 'POST',
      });
      setOrder({ items: [], total: 0 });
    } catch (error) {
      console.error('Error clearing order:', error);
    }
  };

  // Poll for order updates (in real app, use WebSocket)
  useEffect(() => {
    fetchOrder();
    const interval = setInterval(fetchOrder, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, []);

  return {
    order,
    isLoading,
    clearOrder,
    refetch: fetchOrder,
  };
}