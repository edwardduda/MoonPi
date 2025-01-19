import React from "react";
import {
  ChartCanvas,
  Chart,
  CandlestickSeries,
  discontinuousTimeScaleProvider,
  XAxis,
  YAxis,
  OHLCTooltip,
  CrossHairCursor,
  SingleValueTooltip,
  MouseCoordinateX,
  MouseCoordinateY,
} from "react-financial-charts";

const CandlestickChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <div>No data available for candlestick chart.</div>;
  }

  // Ensure step is added to each data point before processing
  const enrichedData = data.map((d, index) => ({
    ...d,
    step: index, // Explicitly add step
    date: new Date(d.date), // Convert date string to Date object
  }));

  const scaleProvider = discontinuousTimeScaleProvider
    .inputDateAccessor(d => d.date);
  
  const { data: chartData, xScale, xAccessor, displayXAccessor } =
    scaleProvider(enrichedData);

  const xExtents = [
    xAccessor(chartData[0]),
    xAccessor(chartData[chartData.length - 1]),
  ];

  // Function to get step from data point
  const stepAccessor = d => d ? d.step : undefined;

  return (
    <ChartCanvas
      height={550}
      ratio={window.devicePixelRatio}
      width={Math.min(1300, window.innerWidth * 0.9)}
      margin={{ left: 50, right: 100, top: 40, bottom: 30 }}
      data={chartData}
      xScale={xScale}
      xAccessor={xAccessor}
      displayXAccessor={displayXAccessor}
      xExtents={xExtents}
    >
      <Chart id={1} yExtents={d => [d.high, d.low]}>
        <XAxis />
        <YAxis />
        <CandlestickSeries />
        <OHLCTooltip origin={[-40, 0]} />
        <MouseCoordinateX
          at="bottom"
          orient="bottom"
          displayFormat={d => {
            const index = Math.floor(xAccessor(d));
            const dataPoint = chartData[index];
            return `Step: ${dataPoint ? dataPoint.step : 'N/A'}`;
          }}
        />
        <MouseCoordinateY
          at="right"
          orient="right"
          displayFormat={format => d => d.toFixed(2)}
        />
        <SingleValueTooltip
          yAccessor={stepAccessor}
          yLabel="Step"
          yDisplayFormat={d => d !== undefined ? d.toString() : 'N/A'}
          origin={[-40, 25]}
        />
        <CrossHairCursor />
      </Chart>
    </ChartCanvas>
  );
};

export default CandlestickChart;