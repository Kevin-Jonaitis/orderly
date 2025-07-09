import { useState, useEffect } from 'react';
import { Container, Navbar, Nav, Row, Col, Card, Image } from 'react-bootstrap';
import { WebRTCAudioRecorder } from './components/WebRTCAudioRecorder';
import { OrderDisplay } from './components/OrderDisplay';
import { MenuUpload } from './components/MenuUpload';

type Page = 'order' | 'menu';

function CurrentMenuImage() {
  const [hasImage, setHasImage] = useState<boolean>(false);

  useEffect(() => {
    const checkMenuImage = async () => {
      try {
        const response = await fetch('http://localhost:8002/api/current-menu-image');
        const data = await response.json();
        setHasImage(data.image !== null);
      } catch (error) {
        console.error('Failed to check menu image:', error);
        setHasImage(false);
      }
    };

    checkMenuImage();
    // Refresh every 5 seconds to check for updates
    const interval = setInterval(checkMenuImage, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!hasImage) {
    return (
      <Card className="mt-3">
        <Card.Body>
          <Card.Title>Current Menu</Card.Title>
          <Card.Text className="text-muted">No menu image uploaded yet.</Card.Text>
        </Card.Body>
      </Card>
    );
  }

  return (
    <div className="mt-3">
      <h4>Current Menu</h4>
      <Image 
        src="http://localhost:8002/menus/menu.jpg" 
        alt="Current Menu"
        fluid
        className="w-100"
        style={{ 
          width: '100%', 
          height: 'auto',
          objectFit: 'cover',
          objectPosition: 'center'
        }}
      />
    </div>
  );
}

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
          <>
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
            <CurrentMenuImage />
          </>
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