# The project, explained simply

## The one-line version

"I built a tool that tells a bond fund how much money it would lose or make if Bank Negara changes interest rates, and puts it all on one dashboard."

## The story version

**1. What's a bond?**
A bond is an IOU. You lend the government RM100, and it pays you a little "thank you" money every six months (the coupon), then gives your RM100 back at the end.

**2. Why do bond prices move?**
Imagine you hold an IOU that pays 4%. Tomorrow the government starts selling new IOUs that pay 5%. Nobody wants your 4% IOU at full price any more, so its price drops.
**Rates up, bond prices down.** That's the most important rule.

**3. What's the OPR?**
It's the interest rate Bank Negara sets for the whole country, the "master dial". When BNM turns the dial up, bond prices fall. When it turns the dial down, they rise.

**4. What's the yield curve?** (`curve.py`)
Short IOUs (3 years) and long IOUs (10 years) pay different rates. Put the rates on a chart from short to long and join the dots, and you get a line called the yield curve. My tool draws the smoothest correct line through Bank Negara's real numbers, using a standard method called Nelson-Siegel.

**5. How much does each bond wobble?** (`bond.py`)
Some bonds wobble a lot when rates move, some only a little. Long bonds wobble more, like a longer seesaw. The tool measures each bond's wobble (**duration**) and how much money moves for each tiny 0.01% change in rates (**DV01**).
> For the sample fund, every 0.01% rise in rates costs about **RM34,600**.

**6. The "what if" machine** (`stress.py`)
This asks: what if BNM raises the OPR by 0.25%? By 1%? What if only long rates rise? It recalculates every bond and gives a ringgit answer.
> A 0.25% OPR hike costs the sample fund about **RM424,000**.

It also shows **where** the risk sits. For this fund, most of it is in the 5-year bonds.

**7. Can a computer predict rates?** (`forecast.py`)
I tested machine learning against the laziest guess possible: "rates tomorrow = rates today". It's a fair test, because each model only sees the past, never the future. The lazy guess is surprisingly hard to beat. Showing that honestly is the point.

**8. The AI reader** (`mps.py`)
Bank Negara publishes a statement after every rate decision. The AI reads it and says: did they hold, hike or cut, and are they leaning towards a hike? Then **my maths, not the AI**, works out what that means for the fund in ringgit. The AI never makes up numbers.

**9. The automatic report** (`factsheet.py`)
With one click, it makes a one-page PDF report for the fund, the kind that's usually built by hand in Excel every month.

**10. The dashboard** (`api/` + `web/`)
Everything shows up on a website. There's a slider: drag the OPR up or down and watch the curve move and the profit or loss change instantly.

## How the pieces connect

```
Bank Negara data → yield curve → value each bond → "what if" tests → dashboard + PDF
                                                  ↑
                         AI reads BNM's statement ┘
```

## 30-second pitch for an interview

"Opus manages Malaysian bonds and sukuk, so I built the tool a fund like that needs. It takes Bank Negara's yield data, fits the MGS curve, and measures the portfolio's interest-rate risk: duration, DV01, and where on the curve the risk sits. Then it stress-tests OPR moves. For example, a 25 basis point hike costs the sample fund about RM424k. It also backtests ML forecasts honestly against a random walk, uses an LLM to read BNM policy statements while keeping all the numbers in deterministic code, and generates the monthly factsheet automatically. The back end is FastAPI and the dashboard is React."

## Questions you might get, with answers

- **"Why does a 1% cut gain more than a 1% hike loses?"** That's convexity: the price-yield line curves, so prices rise faster than they fall.
- **"Why doesn't the long end move as much as the OPR?"** Long yields also depend on expected inflation and growth over many years, not just today's policy rate. The 40% figure is my assumption, and with history I'd estimate it from past OPR decision days.
- **"Why not let the AI do the maths?"** LLMs can make up numbers. In finance, every figure has to be traceable, so the AI only reads text and code does the maths.
- **"Did ML beat the random walk?"** Answer honestly with the real backtest numbers. Short-term yields are close to a random walk, and knowing that is the point.
