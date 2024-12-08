import React, { useState, useEffect } from "react";
import axios from "axios";
import Hls from "hls.js";
import "./App.css";
import AnimatedCanvas from "./assets/AnimatedCanvas ";

function App() {
  const [showSplash, setShowSplash] = useState(true); // Управление заставкой
  const [videoFile, setVideoFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [hlsPath, setHlsPath] = useState("");
  const [analysisResults, setAnalysisResults] = useState(null);
  const [notification, setNotification] = useState(null);
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    // Показ заставки на 
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!videoFile) {
      showNotification("Пожалуйста, выберите видеофайл!", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);

    setIsUploading(true);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.data.hls_path) {
        setHlsPath(response.data.hls_path);
        showNotification("Видео успешно загружено!", "success");
      }
      if (response.data.analysis) {
        setAnalysisResults(response.data.analysis);
      }
    } catch (error) {
      console.error("Ошибка при отправке видео:", error);
      showNotification("Произошла ошибка при загрузке видео.", "error");
    } finally {
      setIsUploading(false);
    }
  };

  const showNotification = (message, type) => {
    setNotification({ message, type });
    setProgress(100);
    setTimeout(() => {
      let interval = setInterval(() => {
        setProgress((prev) => {
          if (prev <= 0) {
            clearInterval(interval);
            setNotification(null);
            return 0;
          }
          return prev - 5;
        });
      }, 100);
    }, 100);
  };

  useEffect(() => {
    if (hlsPath) {
      const video = document.getElementById("video");
      if (Hls.isSupported()) {
        const hls = new Hls();
        hls.loadSource(`http://127.0.0.1:5000/stream/${hlsPath}`);
        hls.attachMedia(video);
      } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = `http://127.0.0.1:5000/stream/${hlsPath}`;
      }
    }
  }, [hlsPath]);



  return (
    <div className="App">
      <AnimatedCanvas/>

       <h1 className="fixed-title">Программа для анализа дорожного трафика</h1>
      {showSplash ? (
        <div className="splash">
          <svg
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            xmlnsXlink="http://www.w3.org/1999/xlink"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
          >
            <path opacity="0.3">
              <animate
                attributeName="d"
                dur="1s"
                repeatCount="indefinite"
                values="M12 2 Q9 2.2 7.3 7 Q6.5 9.5 6.5 12 Q6.5 15 7.3 17 Q9 22 12 22;M12 2 Q6.5 2.2 3.3 7 Q2 9.5 2 12 Q2 15 3.4 17 Q6.5 22 12 22;"
              />
            </path>
            <path opacity="0.3">
              <animate
                attributeName="d"
                dur="1s"
                repeatCount="indefinite"
                values="M12 2 Q12 2.2 12 7 Q12 9.5 12 12 Q12 15 12 17 Q12 22 12 22;M12 2 Q9 2.2 7.3 7 Q6.5 9.5 6.5 12 Q6.5 15 7.3 17 Q9 22 12 22;"
              />
            </path>
            <path opacity="0.3">
              <animate
                attributeName="d"
                dur="1s"
                repeatCount="indefinite"
                values="M12 2 Q15 2.2 16.6 7 Q17.5 9.5 17.5 12 Q17.5 15 16.7 17 Q15 22 12 22;M12 2 Q12 2.2 12 7 Q12 9.5 12 12 Q12 15 12 17 Q12 22 12 22;"
              />
            </path>
            <path opacity="0.3">
              <animate
                attributeName="d"
                dur="1s"
                repeatCount="indefinite"
                values="M12 2 Q17.5 2.2 20.7 7 Q22 9.5 22 12 Q22 15 20.6 17 Q17.5 22 12 22;M12 2 Q15 2.2 16.6 7 Q17.5 9.5 17.5 12 Q17.5 15 16.7 17 Q15 22 12 22;"
              />
            </path>
            <circle cx="12" cy="12" r="10" />
            <path d="M2.4 8.6 Q6 7.1 12 7 Q18 7.1 21.6 8.6" />
            <path d="M2.4 15.2 Q6 17.1 12 17.2 Q17 17.1 21.6 15.2" />
          </svg>
        </div>
      ) : ( <div className="main-content">
       
          <main>
            <form className="upload-form" onSubmit={handleSubmit}>
              <input className="file-input"
                type="file"
                accept="video/mp4, video/avi, video/mov"
                onChange={handleFileChange}
                required
              />
              <button type="submit" disabled={isUploading}>
                {isUploading ? "Загрузка..." : "Отправить Видео"}
              </button>
            </form>

     

{hlsPath && !isUploading && (
  <div className="video-container">
    {isUploading ? (
      <div className="loader"></div> /* Показ загрузочного индикатора */
    ) : hlsPath ? (
      <video id="video" controls autoPlay>  </video> /* Показ видео */
      
    ) : (
      <p style={{ color: "#fff" }}>Видео еще не загружено</p> /* Сообщение, если видео отсутствует */
    )}
  </div>
)}
          </main>

          {notification && (
            <div className={`notification ${notification.type}`}>
              <p>{notification.message}</p>
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
