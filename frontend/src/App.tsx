import { useState, useEffect } from 'react';
import { Container, Navbar, Nav, Row, Col, Card, Image } from 'react-bootstrap';
import { WebRTCAudioRecorder } from './components/WebRTCAudioRecorder';
import { OrderDisplay } from './components/OrderDisplay';
import { MenuUpload } from './components/MenuUpload';

type Page = 'order' | 'menu';

function CurrentMenuImage({ currentPage, refreshKey }: { currentPage: Page, refreshKey: number }) {
  const [hasImage, setHasImage] = useState<boolean>(false);
  const [imageTimestamp, setImageTimestamp] = useState<number>(Date.now());

  useEffect(() => {
    const checkMenuImage = async () => {
      try {
        const response = await fetch('http://localhost:8002/api/current-menu-image');
        const data = await response.json();
        const newHasImage = data.image !== null;
        setHasImage(newHasImage);
        // Update timestamp to force image refresh
        if (newHasImage) {
          setImageTimestamp(Date.now());
        }
      } catch (error) {
        console.error('Failed to check menu image:', error);
        setHasImage(false);
      }
    };

    // Check immediately when page changes to 'order' or refreshKey changes
    if (currentPage === 'order') {
      checkMenuImage();
    }
    
    // Refresh every 5 seconds to check for updates (only when on order page)
    const interval = setInterval(() => {
      if (currentPage === 'order') {
        checkMenuImage();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [currentPage, refreshKey]);

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
        src={`http://localhost:8002/menus/menu.jpg?t=${imageTimestamp}`}
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
  const [menuRefreshKey, setMenuRefreshKey] = useState<number>(0);

  const handleMenuUploaded = () => {
    setMenuRefreshKey(prev => prev + 1);
  };

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
                <WebRTCAudioRecorder />
              </Col>
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <OrderDisplay />
                  </Card.Body>
                </Card>
              </Col>
            </Row>
            <CurrentMenuImage currentPage={currentPage} refreshKey={menuRefreshKey} />
          </>
        ) : (
          <Row>
            <Col lg={8} className="mx-auto">
              <Card>
                <Card.Body>
                  <MenuUpload onMenuUploaded={handleMenuUploaded} />
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