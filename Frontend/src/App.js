// src/App.jsx
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LandingPage from './pages/AppLanding';
import DetectHardwarePage from './pages/DetectHardwarePage';
import FinetuneSettings from './pages/FinetuningSettingsPage';
import './index.css';

const RedirectToFastAPI = () => {
  useEffect(() => {
    window.location.href = "http://127.0.0.1:8000/";
  }, []);

  return null;
};

function App() {
  const [finetuneSettings, setFinetuneSettings] = useState({
    task: 'text-generation',
    model_name: 'llama2-7b',
    compute_specs: 'Standard GPU',
    num_train_epochs: 3,
    lora_r: 16,
    max_seq_length: 512,
    use_4bit: true,
    load_in_8bit: false,
    bnb_4bit_compute_dtype: 'nf4',
    bnb_4bit_use_quant_type: false,
    use_nested_quant: false,
    bnb_4bit_quant_type: 'nf4',
    fp16: false,
    bf16: false,
    per_device_train_batch_size: 2,
    per_device_eval_batch_size: 2,
    learning_rate: 2e-4,
    gradient_accumulation_steps: 1,
    max_grad_norm: 0.3,
    warmup_ratio: 0.03,
    weight_decay: 0.001,
    gradient_checkpointing: false,
    group_by_length: false,
    packing: false,
    optim: 'paged_adamw_32bit',
    lr_scheduler_type: 'cosine',
    max_steps: -1,
    hardware_config: { 
      gpu: '',
      ram: 0,
      disk: 0,
      cpu_cores: 0,
    },
  });
  useEffect(() => {
    console.log('Settings updated in App.js:', finetuneSettings);
  }, [finetuneSettings]);
  // Function to update settings from child components
  const updateSettings = (newSettingsOrFunction) => {
    // Log what we're trying to update with
    console.log('Update triggered with:', 
      typeof newSettingsOrFunction === 'function' 
        ? 'Function updater' 
        : newSettingsOrFunction
    );
    
    setFinetuneSettings(prevSettings => {
      // Handle both direct objects and function updates
      const updatedSettings = typeof newSettingsOrFunction === 'function'
        ? newSettingsOrFunction(prevSettings)
        : { ...prevSettings, ...newSettingsOrFunction };
      
      // Create a completely new object to ensure state update triggers
      const newState = JSON.parse(JSON.stringify(updatedSettings));
      
      console.log('Updating settings in App.js:', newState);
      return newState;
    });
  };

  return (
    <Router>
      <div className="bg-gray-900 text-white min-h-screen">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route 
            path="/finetune/load_settings" 
            element={
              <FinetuneSettings 
                defaultValues={finetuneSettings}
                updateSettings={updateSettings}
              />
            } 
          />
          <Route 
            path="/finetune/detect" 
            element={
              <DetectHardwarePage 
                currentSettings={finetuneSettings}
                updateSettings={updateSettings}
              />
            } 
          />
          {/* <Route
            path="/app"
            element={<DetectHardwarePage />}
          /> */}
          <Route
            path="/app"
            element={<RedirectToFastAPI />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;