{
  "rounds": 50000,
  "price": 100.0,
  "lambda_value": 0.0000667,
  "market_maker": true,
  "record_agent_series": false,
  "seed": 1234,

  "base_cohorts": [
    {
      "memory": 5,
      "strategies": 2,
      "payoff": "BinaryMG",
      "count": 150,
      "position_limit": 0
    },
    {
      "memory": 7,
      "strategies": 4,
      "payoff": "DollarGamePayoff",
      "count": 150,
      "position_limit": 0
    }
  ],

  "vary": "payoff_share",
  "values": [0.0, 0.25, 0.5, 0.75, 1.0],
  "target_payoff": "DollarGamePayoff",
  "target_tag": null
}

