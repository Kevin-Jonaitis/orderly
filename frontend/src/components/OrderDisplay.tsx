import { Button, Badge, Alert, ListGroup } from 'react-bootstrap';
import { useOrder } from '../hooks/useOrder';

export function OrderDisplay() {
  const { order, isLoading, clearOrder } = useOrder();

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h4 className="mb-0">Current Order</h4>
        <Button
          onClick={clearOrder}
          disabled={order.items.length === 0}
          variant="danger"
          size="sm"
        >
          Clear Order
        </Button>
      </div>

      {isLoading && (
        <Alert variant="info" className="text-center">
          Loading...
        </Alert>
      )}

      {order.items.length === 0 ? (
        <Alert variant="light" className="text-center">
          <p className="mb-1">No items in your order yet.</p>
          <small className="text-muted">Start speaking to add items!</small>
        </Alert>
      ) : (
        <>
          <ListGroup className="mb-3">
            {order.items.map((item) => (
              <ListGroup.Item key={item.id} className="d-flex justify-content-between align-items-center">
                <div className="d-flex align-items-center gap-2">
                  <span className="fw-bold">{item.name}</span>
                  {item.quantity > 1 && (
                    <Badge bg="primary">x{item.quantity}</Badge>
                  )}
                </div>
                <span className="fw-bold text-success">
                  ${(item.price * item.quantity).toFixed(2)}
                </span>
              </ListGroup.Item>
            ))}
          </ListGroup>
          
          <Alert variant="secondary" className="text-end mb-0">
            <strong>Total: ${order.total.toFixed(2)}</strong>
          </Alert>
        </>
      )}
    </div>
  );
}