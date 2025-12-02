/**
 * SideChart - Reusable D3.js Line Chart Component for Timeline Synchronization
 *
 * Creates a multi-line chart that can be positioned on either side of a vertical timeline
 * and synchronized with the timeline's zoom/brush interactions.
 */
class SideChart {
  /**
   * @param {Object} params - Configuration object
   * @param {d3.Selection} params.container - SVG group or selection to draw in
   * @param {Array} params.data - Array of data objects with date and value columns
   * @param {Array} params.columns - Array of column names to plot as lines
   * @param {Object} params.options - Chart configuration options
   * @param {number} params.options.width - Width of the chart area
   * @param {number} params.options.height - Height of the chart area
   * @param {string} params.options.side - 'left' or 'right' positioning
   * @param {Object} params.options.colors - Object mapping column names to color hex codes
   * @param {d3.scaleTime} params.options.initialYScale - Initial time scale (Y-axis for vertical timeline)
   * @param {number} params.options.xPosition - X position for the chart
   * @param {number} params.options.yTop - Top Y position
   * @param {number} params.options.yBottom - Bottom Y position
   */
  constructor({ container, data, columns, options }) {
    this.container = container;
    this.data = data;
    this.columns = columns;
    this.width = options.width;
    this.height = options.height;
    this.side = options.side;
    this.colors = options.colors;
    this.yScale = options.initialYScale.copy(); // Time scale (vertical)
    this.xPosition = options.xPosition;
    this.yTop = options.yTop;
    this.yBottom = options.yBottom;

    // Create chart group
    this.chartGroup = this.container.append('g')
      .attr('class', `side-chart-${this.side}`);

    // Find max value across all columns for X scale
    this.maxValue = d3.max(this.data, d =>
      d3.max(this.columns, col => d[col])
    );

    // Gap from the central timeline (padding)
    const gap = 20;

    // Create X scale (horizontal - for values)
    // MIRRORED EFFECT: Both charts have 0 near the center timeline
    // Left chart: 0 at RIGHT (near timeline), max at LEFT (inverted range)
    // Right chart: 0 at LEFT (near timeline), max at RIGHT
    if (this.side === 'left') {
      this.xScale = d3.scaleLinear()
        .domain([0, this.maxValue * 1.1]) // Add 10% padding
        .range([this.xPosition - gap, this.xPosition - this.width]); // Inverted: 0 → near timeline, max → far left
    } else {
      this.xScale = d3.scaleLinear()
        .domain([0, this.maxValue * 1.1])
        .range([this.xPosition + gap, this.xPosition + this.width]); // Normal: 0 → near timeline, max → far right
    }

    // Create line generator
    this.lineGenerator = d3.line()
      .defined(d => d.value !== null && d.value !== undefined && !isNaN(d.value))
      .x(d => this.xScale(d.value))
      .y(d => this.yScale(d.date))
      .curve(d3.curveMonotoneY); // Smooth curves

    // Initialize the chart
    this.render();
  }

  /**
   * Initial render of the chart
   */
  render() {
    // Create clip path to prevent line overflow
    const clipId = `clip-${this.side}-${Math.random().toString(36).substr(2, 9)}`;
    const defs = this.container.append('defs');
    defs.append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('x', this.side === 'left' ? this.xPosition - this.width : this.xPosition)
      .attr('y', this.yTop)
      .attr('width', this.width)
      .attr('height', this.yBottom - this.yTop);

    // Draw background area
    this.chartGroup.append('rect')
      .attr('class', 'chart-background')
      .attr('x', this.side === 'left' ? this.xPosition - this.width : this.xPosition)
      .attr('y', this.yTop)
      .attr('width', this.width)
      .attr('height', this.yBottom - this.yTop)
      .attr('fill', '#fafafa')
      .attr('opacity', 0.5);

    // Create groups for axes and lines
    this.axisGroup = this.chartGroup.append('g').attr('class', 'axis-group');
    this.linesGroup = this.chartGroup.append('g')
      .attr('class', 'lines-group')
      .attr('clip-path', `url(#${clipId})`); // Apply clipping to lines
    this.legendGroup = this.chartGroup.append('g').attr('class', 'legend-group');

    // Draw axes
    this.drawAxes();

    // Draw lines
    this.drawLines();

    // Draw legend
    this.drawLegend();
  }

  /**
   * Draw X axis (horizontal - for values)
   */
  drawAxes() {
    this.axisGroup.selectAll('*').remove();

    // X axis at the bottom
    const xAxis = d3.axisBottom(this.xScale)
      .ticks(5)
      .tickSize(5)
      .tickFormat(d3.format('.0f'));

    const axisY = this.yBottom + 20;
    const axisX = this.side === 'left' ? this.xPosition - this.width : this.xPosition;

    this.axisGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${axisY})`)
      .call(xAxis)
      .selectAll('text')
      .style('font-size', '10px')
      .style('fill', '#666');

    this.axisGroup.selectAll('.x-axis line, .x-axis path')
      .style('stroke', '#999');

    // Add axis label
    const labelX = this.side === 'left'
      ? this.xPosition - this.width / 2
      : this.xPosition + this.width / 2;

    this.axisGroup.append('text')
      .attr('class', 'axis-label')
      .attr('x', labelX)
      .attr('y', axisY + 35)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', '#666')
      .style('font-weight', '500')
      .text('Search Interest');
  }

  /**
   * Draw the line series
   */
  drawLines() {
    this.linesGroup.selectAll('*').remove();

    // Draw each column as a line
    this.columns.forEach(column => {
      // Prepare data for this line
      const lineData = this.data
        .map(d => ({
          date: d.date,
          value: d[column]
        }))
        .filter(d => d.date && !isNaN(d.value));

      // Draw the line
      this.linesGroup.append('path')
        .datum(lineData)
        .attr('class', `line line-${column.replace(/\s+/g, '-')}`)
        .attr('fill', 'none')
        .attr('stroke', this.colors[column] || '#999')
        .attr('stroke-width', 2)
        .attr('opacity', 0.8)
        .attr('d', this.lineGenerator);
    });
  }

  /**
   * Draw legend
   */
  drawLegend() {
    this.legendGroup.selectAll('*').remove();

    // Position legends at outer edges (away from center timeline)
    const legendX = this.side === 'left'
      ? this.xPosition - this.width + 10  // Far left edge
      : this.xPosition + this.width - 100;  // Far right edge (with space for text)
    const legendY = this.yTop - 40;

    this.columns.forEach((column, i) => {
      const legendItem = this.legendGroup.append('g')
        .attr('class', 'legend-item')
        .attr('transform', `translate(${legendX}, ${legendY + i * 18})`);

      // Color circle
      legendItem.append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', 4)
        .attr('fill', this.colors[column] || '#999');

      // Label
      legendItem.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .style('font-size', '10px')
        .style('fill', '#333')
        .text(column);
    });
  }

  /**
   * Update the chart with a new time scale
   * This is called when the timeline is zoomed/brushed
   *
   * @param {d3.scaleTime} newYScale - Updated time scale from timeline
   */
  update(newYScale) {
    // Update the Y scale
    this.yScale = newYScale.copy();

    // Update line generator with new scale
    this.lineGenerator.y(d => this.yScale(d.date));

    // Redraw lines with smooth transition
    this.columns.forEach(column => {
      const lineData = this.data
        .map(d => ({
          date: d.date,
          value: d[column]
        }))
        .filter(d => d.date && !isNaN(d.value));

      this.linesGroup.select(`.line-${column.replace(/\s+/g, '-')}`)
        .datum(lineData)
        .transition()
        .duration(300)
        .attr('d', this.lineGenerator);
    });
  }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SideChart;
}
