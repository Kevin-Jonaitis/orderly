import { useState } from 'react';
import { Container, Navbar, Nav, Row, Col, Card } from 'react-bootstrap';
import { WebRTCAudioRecorder } from './components/WebRTCAudioRecorder';
import { OrderDisplay } from './components/OrderDisplay';
import { MenuUpload } from './components/MenuUpload';

type Page = 'order' | 'menu';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('order');

  return (
    <>
      <Navbar bg="light" expand="lg" className="mb-4">
        <Container>
          <Navbar.Brand>🤖 AI Order Taker</Navbar.Brand>
          <Nav className="ms-auto">
            <Nav.Link 
              active={currentPage === 'order'}
              onClick={() => setCurrentPage('order')}
            >
              Take Order
            </Nav.Link>
            <Nav.Link 
              active={currentPage === 'menu'}
              onClick={() => setCurrentPage('menu')}
            >
              Menu Upload
            </Nav.Link>
          </Nav>
        </Container>
      </Navbar>

      <Container>
        {currentPage === 'order' ? (
          <Row>
            <Col md={6}>
              <Card>
                <Card.Body>
                  <WebRTCAudioRecorder />
                </Card.Body>
              </Card>
            </Col>
            <Col md={6}>
              <Card>
                <Card.Body>
                  <OrderDisplay />
                </Card.Body>
              </Card>
            </Col>
          </Row>
        ) : (
          <Row>
            <Col lg={8} className="mx-auto">
              <Card>
                <Card.Body>
                  <MenuUpload />
                </Card.Body>
              </Card>
            </Col>
          </Row>
        )}
      </Container>
    </>
  );
}

export default App;