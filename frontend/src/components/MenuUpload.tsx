import { useState } from 'react';
import { Button, Form, Alert, Card } from 'react-bootstrap';

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
    <div>
      <h4 className="mb-1">Menu Upload</h4>
      <p className="text-muted mb-4">
        Upload your restaurant menu as text or image. This will be used to help the AI understand your offerings.
      </p>

      <div className="d-flex flex-column gap-4">
        {/* Text Input Section */}
        <Card>
          <Card.Body>
            <Card.Title>Enter Menu Text</Card.Title>
            <Form>
              <Form.Group className="mb-3">
                <Form.Control
                  as="textarea"
                  rows={8}
                  value={textContent}
                  onChange={handleTextChange}
                  placeholder="Enter your menu items here...

Example:
Cheeseburger - $8.99
Fries - $3.99
Drink - $2.99"
                />
              </Form.Group>
              <Button
                onClick={uploadTextContent}
                disabled={isUploading || !textContent.trim()}
                variant="primary"
              >
                {isUploading ? 'Uploading...' : 'Save Menu Text'}
              </Button>
            </Form>
          </Card.Body>
        </Card>

        <div className="text-center text-muted fw-bold">OR</div>

        {/* File Upload Section */}
        <Card>
          <Card.Body>
            <Card.Title>Upload Menu File</Card.Title>
            <Form>
              <Form.Group className="mb-3">
                <Form.Control
                  id="file-input"
                  type="file"
                  onChange={handleFileSelect}
                  accept=".txt,.png,.jpg,.jpeg,.pdf"
                />
                <Form.Text className="text-muted">
                  Supported: Text files (.txt) and Images (.png, .jpg, .jpeg)
                </Form.Text>
              </Form.Group>
              <Button
                onClick={handleUpload}
                disabled={isUploading || !selectedFile}
                variant="primary"
              >
                {isUploading ? 'Uploading...' : 'Upload File'}
              </Button>
            </Form>
          </Card.Body>
        </Card>
      </div>

      {uploadMessage && (
        <Alert 
          variant={uploadMessage.startsWith('✅') ? 'success' : 'danger'}
          className="mt-4"
        >
          {uploadMessage}
        </Alert>
      )}

      <Alert variant="info" className="mt-4">
        <Alert.Heading>How it works:</Alert.Heading>
        <ul className="mb-0">
          <li>Text files are saved directly to the menu database</li>
          <li>Images are processed with OCR to extract menu text (currently stubbed)</li>
          <li>The AI uses this menu information to understand your offerings when taking orders</li>
        </ul>
      </Alert>
    </div>
  );
}