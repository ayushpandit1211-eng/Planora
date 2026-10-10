railway_demo.csv: 10,000 entirely synthetic railway booking records. Not IRCTC or government data. Fictional train codes and fares; real city names. No real passenger details. Journey dates January through September 2026. Upload with dataset name railway_demo, preview, then commit.

Example SQL:
SELECT origin_station, COUNT(*) AS bookings, SUM(fare_amount) AS booked_fare FROM railway_demo WHERE ticket_status = 'Confirmed' GROUP BY origin_station ORDER BY booked_fare DESC;
