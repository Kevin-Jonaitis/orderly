import { useState } from 'react';
import { AudioRecorder } from './components/AudioRecorder';
import { OrderDisplay } from './components/OrderDisplay';
import { MenuUpload } from './components/MenuUpload';

type Page = 'order' | 'menu';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('order');

  return (
    <div>
      <header style={{ padding: '20px', backgroundColor: '#f0f0f0', borderBottom: '1px solid #ccc' }}>
        <h1>🤖 AI Order Taker</h1>
        <nav>
          <button
            onClick={() => setCurrentPage('order')}
            style={{ 
              marginRight: '10px', 
              padding: '10px 20px',
              backgroundColor: currentPage === 'order' ? '#007bff' : '#e0e0e0',
              color: currentPage === 'order' ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Take Order
          </button>
          <button
            onClick={() => setCurrentPage('menu')}
            style={{ 
              padding: '10px 20px',
              backgroundColor: currentPage === 'menu' ? '#007bff' : '#e0e0e0',
              color: currentPage === 'menu' ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Menu Upload
          </button>
        </nav>
      </header>

      <main style={{ padding: '20px' }}>
        {currentPage === 'order' ? (
          <div style={{ display: 'flex', gap: '20px', maxWidth: '1200px' }}>
            <div style={{ flex: 1, padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
              <AudioRecorder />
            </div>
            <div style={{ flex: 1, padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
              <OrderDisplay />
            </div>
          </div>
        ) : (
          <div style={{ maxWidth: '800px', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
            <MenuUpload />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;