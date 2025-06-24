import React, { useState } from 'react';

export function MenuUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const [textContent, setTextContent] = useState<string>('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setUploadMessage('');
    }
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTextContent(event.target.value);
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload-menu', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (response.ok) {
        setUploadMessage(`✅ ${result.message}`);
        setSelectedFile(null);
        // Reset file input
        const fileInput = document.getElementById('file-input') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
      } else {
        setUploadMessage(`❌ Error: ${result.error}`);
      }
    } catch (error) {
      setUploadMessage(`❌ Upload failed: ${error}`);
    }
  };

  const uploadTextContent = async () => {
    if (!textContent.trim()) {
      setUploadMessage('❌ Please enter some text content');
      return;
    }

    // Create a text file from the content
    const blob = new Blob([textContent], { type: 'text/plain' });
    const file = new File([blob], `menu_${Date.now()}.txt`, { type: 'text/plain' });
    
    await uploadFile(file);
    setTextContent('');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage('❌ Please select a file first');
      return;
    }

    setIsUploading(true);
    setUploadMessage('');

    await uploadFile(selectedFile);
    setIsUploading(false);
  };

  return (
    <div className="menu-upload">
      <h2>Menu Upload</h2>
      <p className="description">
        Upload your restaurant menu as text or image. This will be used to help the AI understand your offerings.
      </p>

      <div className="upload-sections">
        {/* Text Input Section */}
        <div className="upload-section">
          <h3>Enter Menu Text</h3>
          <textarea
            value={textContent}
            onChange={handleTextChange}
            placeholder="Enter your menu items here...&#10;&#10;Example:&#10;Cheeseburger - $8.99&#10;Fries - $3.99&#10;Drink - $2.99"
            className="text-input"
            rows={8}
          />
          <button
            onClick={uploadTextContent}
            disabled={isUploading || !textContent.trim()}
            className="upload-button"
          >
            {isUploading ? 'Uploading...' : 'Save Menu Text'}
          </button>
        </div>

        <div className="divider">OR</div>

        {/* File Upload Section */}
        <div className="upload-section">
          <h3>Upload Menu File</h3>
          <div className="file-input-container">
            <input
              id="file-input"
              type="file"
              onChange={handleFileSelect}
              accept=".txt,.png,.jpg,.jpeg,.pdf"
              className="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              {selectedFile ? selectedFile.name : 'Choose file...'}
            </label>
          </div>
          
          <div className="file-types">
            <small>Supported: Text files (.txt) and Images (.png, .jpg, .jpeg)</small>
          </div>

          <button
            onClick={handleUpload}
            disabled={isUploading || !selectedFile}
            className="upload-button"
          >
            {isUploading ? 'Uploading...' : 'Upload File'}
          </button>
        </div>
      </div>

      {uploadMessage && (
        <div className={`upload-message ${uploadMessage.startsWith('✅') ? 'success' : 'error'}`}>
          {uploadMessage}
        </div>
      )}

      <div className="upload-info">
        <h3>How it works:</h3>
        <ul>
          <li>Text files are saved directly to the menu database</li>
          <li>Images are processed with OCR to extract menu text (currently stubbed)</li>
          <li>The AI uses this menu information to understand your offerings when taking orders</li>
        </ul>
      </div>
    </div>
  );
}