import React, { useState, useRef } from 'react';
import { Upload, X, FileImage, Stethoscope, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import './App.css';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setError(null);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setError(null);
      setResults(null);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const resetUpload = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (result) => {
    if (result === 'Normal') {
      return <CheckCircle className="w-6 h-6 text-emerald-500" />;
    } else {
      return <AlertTriangle className="w-6 h-6 text-orange-500" />;
    }
  };

  const getStatusColor = (result) => {
    if (result === 'Normal') {
      return 'border-emerald-200 bg-emerald-50';
    } else {
      return 'border-orange-200 bg-orange-50';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-xl">
              <Stethoscope className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">LungScan AI</h1>
              <p className="text-gray-600 text-sm">Advanced Lung Disease Detection</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-700 px-6 py-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <Upload className="w-5 h-5 mr-2" />
                  Upload Chest X-Ray
                </h2>
                <p className="text-blue-100 text-sm mt-1">
                  Upload a clear chest X-ray image for analysis
                </p>
              </div>

              <div className="p-6">
                {!imagePreview ? (
                  <div
                    className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileImage className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 font-medium mb-2">
                      Drag and drop your X-ray image here
                    </p>
                    <p className="text-gray-400 text-sm mb-4">or click to browse files</p>
                    <div className="bg-blue-600 text-white px-6 py-2 rounded-lg inline-block hover:bg-blue-700 transition-colors">
                      Choose File
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="w-full h-64 object-cover rounded-xl border-2 border-gray-200"
                      />
                      <button
                        onClick={resetUpload}
                        className="absolute top-2 right-2 bg-red-500 text-white p-1.5 rounded-full hover:bg-red-600 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    
                    <button
                      onClick={analyzeImage}
                      disabled={loading}
                      className="w-full bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-3 px-6 rounded-xl font-semibold hover:from-blue-700 hover:to-indigo-800 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Analyzing...</span>
                        </>
                      ) : (
                        <>
                          <Stethoscope className="w-5 h-5" />
                          <span>Analyze X-Ray</span>
                        </>
                      )}
                    </button>
                  </div>
                )}

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleImageSelect}
                  accept="image/*"
                  className="hidden"
                />

                {error && (
                  <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      <p className="text-red-700 font-medium">{error}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {results ? (
              <>
                {/* Detection Results */}
                <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
                  <div className="bg-gradient-to-r from-emerald-600 to-teal-700 px-6 py-4">
                    <h3 className="text-xl font-semibold text-white">Detection Results</h3>
                  </div>
                  
                  <div className="p-6 space-y-4">
                    <div className={`border-2 rounded-xl p-4 ${getStatusColor(results.pneumonia_result)}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-gray-900">Pneumonia Detection</h4>
                          <p className="text-gray-600 text-sm">Bacterial/Viral infection screening</p>
                        </div>
                        {getStatusIcon(results.pneumonia_result)}
                      </div>
                      <div className="mt-3">
                        <span className="text-lg font-bold text-gray-900">{results.pneumonia_result}</span>
                      </div>
                    </div>

                    <div className={`border-2 rounded-xl p-4 ${getStatusColor(results.lung_cancer_result)}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-gray-900">Lung Cancer Detection</h4>
                          <p className="text-gray-600 text-sm">Malignant lesion screening</p>
                        </div>
                        {getStatusIcon(results.lung_cancer_result)}
                      </div>
                      <div className="mt-3">
                        <span className="text-lg font-bold text-gray-900">{results.lung_cancer_result}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Clinical Recommendations */}
                <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
                  <div className="bg-gradient-to-r from-purple-600 to-pink-700 px-6 py-4">
                    <h3 className="text-xl font-semibold text-white">🏥 Clinical Recommendations</h3>
                  </div>
                  
                  <div className="p-6">
                    <ul className="space-y-3">
                      {results.clinical_recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                          <p className="text-gray-700">{rec}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Lifestyle Recommendations */}
                <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
                  <div className="bg-gradient-to-r from-amber-600 to-orange-700 px-6 py-4">
                    <h3 className="text-xl font-semibold text-white">🧬 Preventive & Lifestyle Recommendations</h3>
                  </div>
                  
                  <div className="p-6">
                    <ul className="space-y-3">
                      {results.lifestyle_recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start space-x-3">
                          <div className="w-2 h-2 bg-amber-500 rounded-full mt-2 flex-shrink-0"></div>
                          <p className="text-gray-700">{rec}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Disclaimer */}
                <div className="bg-gray-50 border border-gray-200 rounded-2xl p-6">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-6 h-6 text-amber-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-2">Important Disclaimer</h4>
                      <p className="text-gray-600 text-sm leading-relaxed">
                        This AI analysis is for informational purposes only and should not replace professional medical diagnosis. 
                        Always consult with qualified healthcare professionals for proper medical evaluation and treatment decisions.
                      </p>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
                <FileImage className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-600 mb-2">Ready for Analysis</h3>
                <p className="text-gray-400">
                  Upload a chest X-ray image to get started with AI-powered disease detection
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;