import { Button, Badge, Alert, ListGroup, Card, Stack, Container } from 'react-bootstrap';
import { useOrder } from '../hooks/useOrder';

export function OrderDisplay() {
  const { order, isLoading, clearOrder } = useOrder();

  return (
    <Container>
      <Stack gap={4}>
        <Stack direction="horizontal" className="justify-content-between align-items-center">
          <Card.Title as="h4">Current Order</Card.Title>
          <Button
            onClick={clearOrder}
            disabled={order.items.length === 0}
            variant="danger"
            size="sm"
          >
            Clear Order
          </Button>
        </Stack>

        {isLoading && (
          <Alert variant="info" className="text-center">
            Loading...
          </Alert>
        )}

        {order.items.length === 0 ? (
          <Alert variant="light" className="text-center">
            <Alert.Heading>No items in your order yet.</Alert.Heading>
            <Card.Text as="small" className="text-muted">
              Start speaking to add items!
            </Card.Text>
          </Alert>
        ) : (
          <Stack gap={3}>
            <ListGroup>
              {order.items.map((item) => (
                <ListGroup.Item key={item.frontendId || item.id} className="d-flex justify-content-between align-items-center">
                  <Stack direction="horizontal" gap={2}>
                    <Card.Text as="span" className="fw-bold mb-0">
                      {item.name}
                    </Card.Text>
                    {item.quantity > 1 && (
                      <Badge bg="primary">x{item.quantity}</Badge>
                    )}
                  </Stack>
                  <Card.Text as="span" className="fw-bold text-success mb-0">
                    ${(item.price * item.quantity).toFixed(2)}
                  </Card.Text>
                </ListGroup.Item>
              ))}
            </ListGroup>
            
            <Alert variant="secondary" className="text-end">
              <Alert.Heading>Total: ${order.total.toFixed(2)}</Alert.Heading>
            </Alert>
          </Stack>
        )}
      </Stack>
    </Container>
  );
}