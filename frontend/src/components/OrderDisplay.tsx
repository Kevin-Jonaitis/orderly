import React from 'react';
import { useOrder } from '../hooks/useOrder';

export function OrderDisplay() {
  const { order, isLoading, clearOrder } = useOrder();

  return (
    <div className="order-display">
      <div className="order-header">
        <h2>Current Order</h2>
        <button
          onClick={clearOrder}
          className="clear-button"
          disabled={order.items.length === 0}
        >
          Clear Order
        </button>
      </div>

      {isLoading && <div className="loading">Loading...</div>}

      <div className="order-items">
        {order.items.length === 0 ? (
          <div className="empty-order">
            <p>No items in your order yet.</p>
            <p className="hint">Start speaking to add items!</p>
          </div>
        ) : (
          <>
            {order.items.map((item) => (
              <div key={item.id} className="order-item">
                <div className="item-info">
                  <span className="item-name">{item.name}</span>
                  {item.quantity > 1 && (
                    <span className="item-quantity">x{item.quantity}</span>
                  )}
                </div>
                <div className="item-price">
                  ${(item.price * item.quantity).toFixed(2)}
                </div>
              </div>
            ))}
            
            <div className="order-total">
              <strong>Total: ${order.total.toFixed(2)}</strong>
            </div>
          </>
        )}
      </div>
    </div>
  );
}