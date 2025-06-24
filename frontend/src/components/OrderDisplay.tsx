import { useOrder } from '../hooks/useOrder';

export function OrderDisplay() {
  const { order, isLoading, clearOrder } = useOrder();

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ margin: 0, color: '#1f2937' }}>Current Order</h2>
        <button
          onClick={clearOrder}
          disabled={order.items.length === 0}
          style={{
            padding: '8px 16px',
            border: 'none',
            backgroundColor: order.items.length === 0 ? '#9ca3af' : '#ef4444',
            color: 'white',
            borderRadius: '4px',
            cursor: order.items.length === 0 ? 'not-allowed' : 'pointer'
          }}
        >
          Clear Order
        </button>
      </div>

      {isLoading && (
        <div style={{ textAlign: 'center', color: '#6b7280', padding: '32px' }}>
          Loading...
        </div>
      )}

      <div>
        {order.items.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#6b7280', padding: '32px' }}>
            <p>No items in your order yet.</p>
            <p style={{ fontSize: '14px', marginTop: '8px' }}>Start speaking to add items!</p>
          </div>
        ) : (
          <>
            {order.items.map((item) => (
              <div key={item.id} style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                padding: '16px', 
                borderBottom: '1px solid #e5e7eb' 
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontWeight: '500', color: '#1f2937' }}>{item.name}</span>
                  {item.quantity > 1 && (
                    <span style={{
                      backgroundColor: '#3b82f6',
                      color: 'white',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      x{item.quantity}
                    </span>
                  )}
                </div>
                <div style={{ fontWeight: '500', color: '#059669' }}>
                  ${(item.price * item.quantity).toFixed(2)}
                </div>
              </div>
            ))}
            
            <div style={{ 
              backgroundColor: '#f3f4f6', 
              padding: '16px', 
              borderRadius: '8px', 
              marginTop: '16px', 
              textAlign: 'right', 
              fontSize: '18px', 
              color: '#1f2937' 
            }}>
              <strong>Total: ${order.total.toFixed(2)}</strong>
            </div>
          </>
        )}
      </div>
    </div>
  );
}