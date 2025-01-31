import React, { useState } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const CombinedChart = ({ data }) => {
  const [showActions, setShowActions] = useState(true);
  const [showPortfolio, setShowPortfolio] = useState(true);
  const [showRewards, setShowRewards] = useState(true);
  const [showSharpe, setShowSharpe] = useState(true);

  // Define chart colors and opacities
  const chartColors = {
    actions: {
      buy: '#22c55e',    // Green
      sell: '#ef4444',   // Red
      hold: '#94a3b8'    // Gray
    },
    portfolio: '#2563eb', // Blue
    rewards: '#8b5cf6',   // Purple
    sharpe: '#f59e0b'    // Amber
  };

  const containerStyle = {
    width: '100%',
    height: '500px',
    marginBottom: '20px',
    padding: '20px 10px'
  };

  const buttonStyle = {
    padding: '8px 16px',
    marginRight: '8px',
    borderRadius: '4px',
    border: 'none',
    cursor: 'pointer'
  };

  const activeButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#2196F3',
    color: 'white'
  };

  const inactiveButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#e2e8f0',
    color: '#4a5568'
  };

  // Custom bar shape for actions (unchanged)
  const CustomBar = (props) => {
    const { x, y, width, height, value, payload } = props;
    let fill;
    switch(value) {
      case 1: fill = chartColors.actions.buy; break;
      case 2: fill = chartColors.actions.sell; break;
      default: fill = chartColors.actions.hold;
    }
    
    return (
      <g>
        <rect x={x} y={y} width={width} height={height} fill={fill} fillOpacity={0.75} />
        {width > 30 && (
          <text
            x={x + width/2}
            y={y - 5}
            textAnchor="middle"
            fill="#666"
            fontSize={10}
          >
            {payload.basedOnState || ''}
          </text>
        )}
      </g>
    );
  };

  return (
    <div style={{ width: '100%' }}>
      <div style={{ marginBottom: '16px' }}>
        <span style={{ marginRight: '12px', fontWeight: '500' }}>Toggle Metrics:</span>
        <button 
          onClick={() => setShowActions(!showActions)}
          style={showActions ? activeButtonStyle : inactiveButtonStyle}
        >
          Actions
        </button>
        <button 
          onClick={() => setShowPortfolio(!showPortfolio)}
          style={showPortfolio ? activeButtonStyle : inactiveButtonStyle}
        >
          Portfolio Value
        </button>
        <button 
          onClick={() => setShowRewards(!showRewards)}
          style={showRewards ? activeButtonStyle : inactiveButtonStyle}
        >
          Rewards
        </button>
        <button 
          onClick={() => setShowSharpe(!showSharpe)}
          style={showSharpe ? activeButtonStyle : inactiveButtonStyle}
        >
          Sharpe Ratio
        </button>
      </div>
      
      <div style={containerStyle}>
        <ResponsiveContainer>
          <ComposedChart 
            data={data}
            margin={{ top: 20, right: 50, left: 50, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="step" 
              type="number"
              domain={['dataMin', 'dataMax']}
              padding={{ left: 20, right: 20 }}
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
            />
            
            <YAxis 
              yAxisId="portfolio"
              orientation="left"
              domain={['auto', 'auto']}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value}`}
              stroke="#6b7280"
            />
            
            <YAxis 
              yAxisId="actions"
              orientation="right"
              domain={[-0.5, 2.5]}
              ticks={[0, 1, 2]}
              tickFormatter={(value) => (['Hold', 'Buy', 'Sell'][value])}
              tick={{ fontSize: 12 }}
              stroke="#6b7280"
            />
            
            <YAxis 
              yAxisId="sharpe"
              orientation="right"
              domain={['auto', 'auto']}
              tick={{ fontSize: 12 }}
              tickCount={5}
              stroke="#f59e0b"
            />
            
            <Tooltip 
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}
              formatter={(value, name, props) => {
                if (name === 'Actions') {
                  return ['Hold', 'Buy', 'Sell'][value];
                }
                if (name === 'Portfolio Value') {
                  return `$${value.toFixed(2)}`;
                }
                if (name === 'Sharpe Ratio') {
                  return value.toFixed(3);
                }
                return value.toFixed(4);
              }}
            />
            <Legend verticalAlign="top" height={36} />

            {showActions && (
              <Bar 
                yAxisId="actions"
                dataKey="action" 
                name="Actions" 
                shape={<CustomBar />}
                isAnimationActive={false}
              />
            )}

            {showPortfolio && (
              <Line
                yAxisId="portfolio"
                type="monotone"
                dataKey="portfolioValue"
                stroke={chartColors.portfolio}
                strokeWidth={2}
                name="Portfolio Value"
                dot={false}
                strokeOpacity={0.9}
              />
            )}

            {showRewards && (
              <Line
                yAxisId="actions"
                type="monotone"
                dataKey="reward"
                stroke={chartColors.rewards}
                strokeWidth={2}
                name="Rewards"
                dot={false}
                strokeOpacity={0.9}
              />
            )}

            {showSharpe && (
              <Line
                yAxisId="sharpe"
                type="monotone"
                dataKey="sharpe_ratio"
                stroke={chartColors.sharpe}
                strokeWidth={2}
                name="Sharpe Ratio"
                dot={false}
                strokeOpacity={0.9}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CombinedChart;