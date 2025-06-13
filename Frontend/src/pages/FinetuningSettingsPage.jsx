import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { config } from '../services/api';

const FinetuneSettings = ({ defaultValues, updateSettings }) => {
  const navigate = useNavigate();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [formState, setFormState] = useState({});
  const [settingsUpdated, setSettingsUpdated] = useState(false);
  const [activeTooltip, setActiveTooltip] = useState(null);
  
  useEffect(() => {
    const fetchDefaultSettings = async () => {
      try {
        const response = await fetch(`${config.baseURL}/finetune/load_settings`);
        if (!response.ok) throw new Error('Failed to fetch settings');
        
        const data = await response.json();
        console.log("Fetched default values:", data.default_values);
        defaultValues = data.default_values;
        // Update form state with fetched values
        setFormState(data.default_values);
      } catch (err) {
        console.error("Error fetching settings:", err);
      }
    };

    fetchDefaultSettings();
  }, []);

  // Sync with props when they change
  useEffect(() => {
    console.log("defaultValues changed in FinetuneSettings:", defaultValues);
    if (defaultValues) {
      // Create a deep copy to break any references
      const values = JSON.parse(JSON.stringify(defaultValues));
      console.log("Setting form values to:", values);
      setFormState(values);
    }
  }, [defaultValues]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    // Create a completely new object for state update
    const updatedState = {
      ...formState,
      [name]: type === 'number' ? Number(newValue) : newValue
    };

    console.log(`Changing ${name} to:`, newValue);
    console.log("New form values:", updatedState);

    // If task or model is changing, log it prominently
    if (name === 'task' || name === 'model_name') {
      console.log(`⚠️ IMPORTANT: ${name} changed to "${newValue}"`);
    }

    setFormState(updatedState);
  };

  const handleQuantizationChange = (value) => {
    const updatedState = {
      ...formState,
      use_4bit: value === '4bit',
      load_in_8bit: value === '8bit',
      quantization: value
    };

    setFormState(updatedState);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Create FormData to send multipart data
    const formData = new FormData();

    // Append the selected file
    if (selectedFile) {
      formData.append("json_file", selectedFile, selectedFile.name);
    }

    // Append settings as a JSON string
    formData.append("settings", JSON.stringify(formState));

    console.log("Sending data to server:",JSON.stringify(formState));

    try {
      const response = await fetch(`${config.baseURL}/finetune/load_settings`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const responseGet = await fetch(`${config.baseURL}/finetune/start`, {
        method: 'GET',
      });
      console.log("Response from GET request:", responseGet);
      setTimeout(() => {
        navigate('/finetune/loading'); // change here 
      }, 1000);

    } catch (error) {
      console.error("Error submitting form:", error);
      alert("Failed to update settings.");
    }
  };

  // Tooltip definitions
  const tooltips = {
    task: "What you want your AI to be good at doing",
    model_name: "The base AI model you're customizing",
    gpu: "The graphics card that will run your training",
    ram: "Computer memory available for training",
    num_train_epochs: "How many times the AI will see your training data",
    learning_rate: "How quickly the model adapts to new information",
    per_device_train_batch_size: "How many examples are processed at once",
    max_seq_length: "Maximum text length the model can handle",
    dataset_file: "Your training examples in JSON format",
    lora_r: "Controls model capacity and training speed",
    lora_alpha: "Controls how much the model changes during training",
    quantization: "Reduces model size to fit in memory",
    bnb_4bit_compute_dtype: "Technical setting for calculation precision",
    optim: "Method used to update the model during training",
    lr_scheduler_type: "How learning speed changes during training"
    model_desc: "A description for your model for you to refer to it later"
  };

  // Tooltip display component
  const Tooltip = ({ id, children }) => (
    <div className="flex justify-between items-center">
      <div className="flex-grow">{children}</div>
      <div className="relative">
        <span 
          className="cursor-help text-orange-500 font-bold text-sm"
          onMouseEnter={() => setActiveTooltip(id)}
          onMouseLeave={() => setActiveTooltip(null)}
        >
          ?
        </span>
        {activeTooltip === id && (
          <div className="absolute z-10 w-64 p-2 text-sm text-white bg-gray-900 rounded-md shadow-lg border border-orange-500 right-0 top-6">
            {tooltips[id]}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Finetuning Settings</h1>
        <p className="text-gray-400 mt-2">Configure your model training parameters</p>
      </div>

      {settingsUpdated && (
        <div className="bg-green-700 text-white p-4 rounded-lg mb-6 flex items-center justify-between">
          <div>Settings updated successfully! Redirecting...</div>
          <div className="animate-spin h-5 w-5 border-2 border-white rounded-full border-t-transparent"></div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Configuration Summary */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Configuration Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Tooltip id="task">
                <label className="block text-sm font-medium text-gray-400 mb-1">Task</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.task || defaultValues.task || 'Not set'}
              </div>
            </div>
            <div>
              <Tooltip id="model_name">
                <label className="block text-sm font-medium text-gray-400 mb-1">Model Name</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.model_name || defaultValues.model_name || 'Not set'}
              </div>
            </div>
            <div>
                <Tooltip id="model_desc">
                    <label className="block text-sm font-medium text-gray-400 mb-1">Model Description</label>
                </Tooltip>
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                    {formState.model_desc || defaultValues.model_desc || 'Set a description for your model'}
                </div>
            </div>
            <div>
              <Tooltip id="gpu">
                <label className="block text-sm font-medium text-gray-400 mb-1">GPU</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.hardware_config?.gpu || defaultValues.hardware_config?.gpu || 'N/A'}
              </div>
            </div>
            <div>
              <Tooltip id="ram">
                <label className="block text-sm font-medium text-gray-400 mb-1">RAM</label>
              </Tooltip>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-white">
                {formState.hardware_config?.ram || defaultValues.hardware_config?.ram ?
                  `${formState.hardware_config?.ram || defaultValues.hardware_config?.ram} GB` : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Basic Settings */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Basic Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <Tooltip id="num_train_epochs">
                <label htmlFor="num_train_epochs" className="block text-sm font-medium text-gray-400 mb-1">
                  Training Epochs
                </label>
              </Tooltip>
              <input
                type="number"
                id="num_train_epochs"
                name="num_train_epochs"
                min="1"
                max="100"
                value={formState.num_train_epochs || 3}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="learning_rate">
                <label htmlFor="learning_rate" className="block text-sm font-medium text-gray-400 mb-1">
                  Learning Rate
                </label>
              </Tooltip>
              <input
                type="number"
                id="learning_rate"
                name="learning_rate"
                step="0.000001"
                value={formState.learning_rate || 0.0002}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="per_device_train_batch_size">
                <label htmlFor="per_device_train_batch_size" className="block text-sm font-medium text-gray-400 mb-1">
                  Batch Size (Train)
                </label>
              </Tooltip>
              <input
                type="number"
                id="per_device_train_batch_size"
                name="per_device_train_batch_size"
                min="1"
                value={formState.per_device_train_batch_size || 2}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div>
              <Tooltip id="max_seq_length">
                <label htmlFor="max_seq_length" className="block text-sm font-medium text-gray-400 mb-1">
                  Max Sequence Length
                </label>
              </Tooltip>
              <input
                type="number"
                id="max_seq_length"
                name="max_seq_length"
                value={formState.max_seq_length || 512}
                onChange={handleInputChange}
                className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
              />
            </div>

            <div className="col-span-full">
              <Tooltip id="dataset_file">
                <label htmlFor="dataset_file" className="block text-sm font-medium text-gray-400 mb-1">
                  Dataset File
                </label>
              </Tooltip>
              <div className="flex items-center">
                <label className="w-full flex items-center justify-center px-4 py-2 border border-gray-700 rounded-lg cursor-pointer bg-gray-900 hover:bg-gray-800 transition">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 mr-2 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <span className="text-white">Upload Dataset</span>
                  <input
                    type="file"
                    id="dataset_file"
                    name="dataset_file"
                    accept=".json,.jsonl"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>
              </div>
              <p className="mt-2 text-sm text-gray-400">
                {selectedFile ? selectedFile.name : 'No file selected'}
              </p>
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="flex justify-center mb-6">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center text-orange-500 hover:text-orange-400 transition"
          >
            <span className="mr-2">Advanced Settings</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className={`h-5 w-5 transform transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {showAdvanced && (
          <div className="space-y-6">
            {/* LoRA Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">LoRA Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Tooltip id="lora_r">
                  <label htmlFor="lora_r" className="block text-sm font-medium text-gray-400 mb-1">
                    LoRA Rank (r)
                  </label>
                </Tooltip>
                <select
                  id="lora_r"
                  name="lora_r"
                  value={formState.lora_r || 16}
                  onChange={handleInputChange}
                  className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none" >
                  <option value="2">4</option>
                  <option value="8">8</option>
                  <option value="16">16</option>
                  <option value="32">32</option>
                </select>
              </div>
              <div>
                <Tooltip id="lora_alpha">
                  <label htmlFor="lora_alpha" className="block text-sm font-medium text-gray-400 mb-1">
                    LoRA Alpha
                  </label>
                </Tooltip>
                <input
                  type="number"
                  id="lora_alpha"
                  name="lora_alpha"
                  min="1"
                  value={formState.lora_alpha || 32}
                  onChange={handleInputChange}
                  className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                />
              </div>
              </div>
            </div>

            {/* Quantization Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">Quantization Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Tooltip id="quantization">
                    <label className="block text-sm font-medium text-gray-400 mb-2">Precision</label>
                  </Tooltip>
                  <div className="space-y-3">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="4bit"
                        checked={formState.quantization === '4bit'}
                        onChange={() => handleQuantizationChange('4bit')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">4-bit Quantization</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="8bit"
                        checked={formState.quantization === '8bit'}
                        onChange={() => handleQuantizationChange('8bit')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">8-bit Quantization</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="quantization"
                        value="none"
                        checked={formState.quantization === 'none'}
                        onChange={() => handleQuantizationChange('none')}
                        className="rounded-full bg-gray-900 border-gray-700 text-orange-500 focus:ring-orange-500"
                      />
                      <span className="ml-2 text-sm text-gray-400">No Quantization</span>
                    </label>
                  </div>
                </div>

                <div>
                  <Tooltip id="bnb_4bit_compute_dtype">
                    <label htmlFor="bnb_4bit_compute_dtype" className="block text-sm font-medium text-gray-400 mb-1">
                      Compute Dtype
                    </label>
                  </Tooltip>
                  <select
                    id="bnb_4bit_compute_dtype"
                    name="bnb_4bit_compute_dtype"
                    value={formState.bnb_4bit_compute_dtype || 'nf4'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="float32">float32</option>
                    <option value="bfloat16">bfloat16</option>
                    <option value="float16">float16</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Optimization Settings */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-medium text-white mb-4">Optimization Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Tooltip id="optim">
                    <label htmlFor="optim" className="block text-sm font-medium text-gray-400 mb-1">
                      Optimizer
                    </label>
                  </Tooltip>
                  <select
                    id="optim"
                    name="optim"
                    value={formState.optim || 'paged_adamw_32bit'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="paged_adamw_32bit">Paged AdamW 32bit</option>
                    <option value="adamw_torch">AdamW Torch</option>
                    <option value="adamw_bnb_8bit">AdamW 8-bit</option>
                  </select>
                </div>
                <div>
                  <Tooltip id="lr_scheduler_type">
                    <label htmlFor="lr_scheduler_type" className="block text-sm font-medium text-gray-400 mb-1">
                      Learning Rate Scheduler
                    </label>
                  </Tooltip>
                  <select
                    id="lr_scheduler_type"
                    name="lr_scheduler_type"
                    value={formState.lr_scheduler_type || 'cosine'}
                    onChange={handleInputChange}
                    className="bg-gray-900 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="cosine">Cosine</option>
                    <option value="linear">Linear</option>
                    <option value="constant">Constant</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="text-center">
          <button
            type="submit"
            className="bg-orange-500 hover:bg-orange-600 px-6 py-3 rounded-lg font-medium transition inline-flex items-center"
          >
            Start Finetuning
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 ml-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </button>
        </div>
      </form>
    </div>
  );
};

export default FinetuneSettings;