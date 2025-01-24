import React, { useState, useEffect } from "react";
import CombinedChart from './CombinedChart';
import CandlestickChart from "./CandlestickChart";

const EpisodeVisualizer = () => {
  const [episodeData, setEpisodeData] = useState([]);
  const [selectedEpisode, setSelectedEpisode] = useState(321);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadEpisodeData();
  }, [selectedEpisode]);

  const loadEpisodeData = async () => {
    try {
      const filePath = `episode_logs/training_episode_${selectedEpisode}.json`;
      console.log("Attempting to fetch from:", filePath);

      const response = await fetch(filePath);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const text = await response.text();
      const jsonData = JSON.parse(text);

      if (!jsonData.steps || !Array.isArray(jsonData.steps)) {
        throw new Error('Invalid episode data format. Missing "steps" array.');
      }

      const formattedData = jsonData.steps.map((step, index) => {
        const previousState = index > 0 ? jsonData.steps[index - 1] : null;
        
        return {
          step: index,
          action: step.actions[0],
          actionLabel: step.actions[0] === 0 ? "Hold" : step.actions[0] === 1 ? "Buy" : "Sell",
          stateInfo: {
            previousPortfolioValue: previousState ? 
              (Array.isArray(previousState.state) ? previousState.state[0][0] : 0) : 0,
            previousPnL: previousState ? 
              (Array.isArray(previousState.state) ? previousState.state[0][1] : 0) : 0,
          },
          q_values: Array.isArray(step.q_values) ? step.q_values[0] : step.q_values,
          reward: Array.isArray(step.rewards) ? step.rewards[0] : step.rewards,
          portfolioValue: Array.isArray(step.state) ? step.state[0][0] : 0,
          pnl: Array.isArray(step.state) ? step.state[0][1] : 0,
          date: step.date || new Date().toISOString(),
          open: step.open || 0,
          high: step.high || 0,
          low: step.low || 0,
          close: step.close || 0,
          timeStep: `t${index}`,
          basedOnState: index > 0 ? `t${index-1}` : null,
          sharpe_ratio: step.sharpe_ratio || 0,
        };
      });

      setEpisodeData(formattedData);
      setError(null);

    } catch (err) {
      console.error("Error loading episode data:", err);
      setError(`Failed to load episode data: ${err.message}`);
      setEpisodeData([]);
    }
  };

  const handleEpisodeChange = (e) => {
    const episode = parseInt(e.target.value, 10);
    if (!isNaN(episode) && episode > 0) {
      setSelectedEpisode(episode);
    }
  };

  const handlePrevEpisode = () => {
    if (selectedEpisode > 1) {
      setSelectedEpisode(selectedEpisode - 1);
    }
  };

  const handleNextEpisode = () => {
    setSelectedEpisode(selectedEpisode + 1);
  };

  const handleChartClick = () => {
    loadEpisodeData();
  };

  if (error) {
    return (
      <div className="p-4 bg-red-100 border border-red-500 rounded-md m-4">
        <strong>Error:</strong> {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col justify-center items-center p-4 w-full h-screen">
      <div className="bg-white rounded-lg shadow-sm p-5 max-w-7xl w-11/12">
        <h2 className="text-2xl font-bold mb-4">
          
          Episode {selectedEpisode} Performance Dashboard
        </h2>

        {/* Enhanced Episode Selector */}
        <div className="flex items-center gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
          <button 
            onClick={handlePrevEpisode}
            disabled={selectedEpisode <= 1}
            className="p-2 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className="w-4 h-4 border-t-2 border-l-2 border-gray-600 transform -rotate-45" />
          </button>
          
          <div className="flex items-center gap-3">
            <label htmlFor="episodeInput" className="font-medium">
              Episode:
            </label>
            <input
              type="number"
              id="episodeInput"
              value={selectedEpisode}
              onChange={handleEpisodeChange}
              className="w-24 p-2 border border-gray-200 rounded-md text-center"
              min="1"
            />
          </div>

          <button 
            onClick={handleNextEpisode}
            className="p-2 rounded-lg bg-white border border-gray-200 hover:bg-gray-50"
          >
            <div className="w-4 h-4 border-t-2 border-r-2 border-gray-600 transform rotate-45" />
          </button>
        </div>

        {/* Candlestick Chart */}
        <ChartWrapper title="Price Action (Candlestick Chart)">
          <div onClick={handleChartClick} className="cursor-pointer">
            <CandlestickChart
              data={episodeData.map((step) => ({
                date: step.date,
                open: step.open,
                high: step.high,
                low: step.low,
                close: step.close,
                step: step.step
              }))}
            />
          </div>
        </ChartWrapper>

        <ChartWrapper title="Trading Performance">
          <CombinedChart data={episodeData} />
        </ChartWrapper>
      </div>
    </div>
  );
};

const ChartWrapper = ({ title, children }) => (
  <div className="h-[550px] mb-4 flex flex-col items-center">
    <h3 className="text-lg font-semibold mb-2 w-full">{title}</h3>
    <div className="w-full flex justify-center">
      {children}
    </div>
  </div>
);

export default EpisodeVisualizer;