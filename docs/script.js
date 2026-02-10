// Cumulative Distribution Function for Standard Normal Distribution
function normalCDF(x) {
    const t = 1 / (1 + .2316419 * Math.abs(x));
    const d = .3989423 * Math.exp(-x * x / 2);
    let prob = d * t * (.3193815 + t * (-.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if (x > 0) prob = 1 - prob;
    return prob;
}

function blackScholes(S, K, T, r, sigma, type) {
    if (T <= 0) {
        // Intrinsic value
        if (type === 'call') return Math.max(S - K, 0);
        else return Math.max(K - S, 0);
    }
    var d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    var d2 = d1 - sigma * Math.sqrt(T);

    if (type === 'call') {
        return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
    } else {
        return K * Math.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const inputs = ['S', 'K', 'T', 'r', 'sigma'];
    const elements = {};

    inputs.forEach(id => {
        elements[id] = document.getElementById(id);
        elements[id].addEventListener('input', updateResults);
    });

    const ctx = document.getElementById('payoffChart').getContext('2d');
    let chart;

    function updateResults() {
        const S = parseFloat(elements['S'].value);
        const K = parseFloat(elements['K'].value);
        const T = parseFloat(elements['T'].value);
        const r = parseFloat(elements['r'].value);
        const sigma = parseFloat(elements['sigma'].value);

        const callPrice = blackScholes(S, K, T, r, sigma, 'call');
        const putPrice = blackScholes(S, K, T, r, sigma, 'put');

        document.getElementById('call-price').innerText = callPrice.toFixed(4);
        document.getElementById('put-price').innerText = putPrice.toFixed(4);

        updateChart(S, K, T, r, sigma);
    }

    function updateChart(S, K, T, r, sigma) {
        // Generate data for chart: Option Price vs Spot Price
        const spots = [];
        const callPrices = [];
        const putPrices = [];

        // Range: +/- 50% of S
        const minS = S * 0.5;
        const maxS = S * 1.5;
        const steps = 50;
        const stepSize = (maxS - minS) / steps;

        for (let i = 0; i <= steps; i++) {
            const currentS = minS + i * stepSize;
            spots.push(currentS.toFixed(2));
            callPrices.push(blackScholes(currentS, K, T, r, sigma, 'call'));
            putPrices.push(blackScholes(currentS, K, T, r, sigma, 'put'));
        }

        if (chart) {
            chart.data.labels = spots;
            chart.data.datasets[0].data = callPrices;
            chart.data.datasets[1].data = putPrices;
            chart.update();
        } else {
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: spots,
                    datasets: [{
                        label: 'Call Price',
                        data: callPrices,
                        borderColor: 'blue',
                        fill: false
                    }, {
                        label: 'Put Price',
                        data: putPrices,
                        borderColor: 'red',
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Spot Price'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Option Price'
                            }
                        }
                    }
                }
            });
        }
    }

    // Initial calculation
    updateResults();
});
