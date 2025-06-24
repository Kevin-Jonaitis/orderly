import React, { useState } from 'react';
import { AudioRecorder } from './components/AudioRecorder';
import { OrderDisplay } from './components/OrderDisplay';
import { MenuUpload } from './components/MenuUpload';
import './App.css';

type Page = 'order' | 'menu';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('order');

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 AI Order Taker</h1>
        <nav className="navigation">
          <button
            className={`nav-button ${currentPage === 'order' ? 'active' : ''}`}
            onClick={() => setCurrentPage('order')}
          >
            Take Order
          </button>
          <button
            className={`nav-button ${currentPage === 'menu' ? 'active' : ''}`}
            onClick={() => setCurrentPage('menu')}
          >
            Menu Upload
          </button>
        </nav>
      </header>

      <main className="app-main">
        {currentPage === 'order' ? (
          <div className="order-page">
            <div className="order-section">
              <AudioRecorder />
            </div>
            <div className="order-section">
              <OrderDisplay />
            </div>
          </div>
        ) : (
          <div className="menu-page">
            <MenuUpload />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
