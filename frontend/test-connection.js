// Simple test to verify backend connection
const API_BASE = 'http://localhost:9001';

async function testConnection() {
  console.log('🔍 Testing connection to:', API_BASE);
  
  try {
    // Test health endpoint
    console.log('Testing /health endpoint...');
    const healthResponse = await fetch(`${API_BASE}/health`);
    console.log('Health response:', healthResponse.status, healthResponse.statusText);
    const healthData = await healthResponse.json();
    console.log('Health data:', healthData);
    
    // Test chat endpoint
    console.log('\nTesting /v1/chat/completions endpoint...');
    const chatResponse = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'Hello, test message!' }],
        session_id: 'test',
        temperature: 0.7,
        max_tokens: 50,
        top_p: 0.9,
        stream: false,
      }),
    });
    
    console.log('Chat response:', chatResponse.status, chatResponse.statusText);
    const chatData = await chatResponse.json();
    console.log('Chat data:', JSON.stringify(chatData, null, 2));
    
    if (chatData.choices && chatData.choices[0] && chatData.choices[0].message) {
      console.log('✅ Assistant response:', chatData.choices[0].message.content);
    }
    
  } catch (error) {
    console.error('❌ Connection test failed:', error);
  }
}

testConnection();