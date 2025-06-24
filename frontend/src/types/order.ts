export interface OrderItem {
  id: string; // Backend product ID
  name: string;
  price: number;
  quantity: number;
  frontendId?: string; // Frontend unique key for React
}

export interface Order {
  items: OrderItem[];
  total: number;
}

export interface AudioMessage {
  type: 'transcription' | 'audio_response' | 'error';
  text?: string;
  message?: string;
}