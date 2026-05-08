-- 6-patient subset of an EHR fixture for the e2e test.

INSERT INTO patients (patient_id, name, dob, status) VALUES
    ('P001', 'Jane Smith',    '1985-03-15', 'active'),
    ('P002', 'Marcus Lee',    '1972-11-04', 'active'),
    ('P003', 'Aisha Khan',    '1990-07-22', 'active'),
    ('P004', 'Rafael Ortiz',  '1958-09-30', 'active'),
    ('P005', 'Priya Iyer',    '1995-12-01', 'active'),
    ('P006', 'Daniel Park',   '1980-04-19', 'active');

INSERT INTO allergies (patient_id, substance) VALUES
    ('P001', 'penicillin'),
    ('P003', 'sulfa'),
    ('P005', 'latex'),
    ('P005', 'shellfish');

INSERT INTO medications (patient_id, name, dose) VALUES
    ('P001', 'lisinopril',     '10mg daily'),
    ('P003', 'levothyroxine',  '50mcg daily'),
    ('P004', 'metformin',      '1000mg twice daily'),
    ('P004', 'atorvastatin',   '20mg nightly'),
    ('P006', 'sertraline',     '100mg daily');

INSERT INTO diagnoses (patient_id, code, description, date) VALUES
    ('P001', 'I10',    'Essential hypertension',                     '2024-06-12'),
    ('P003', 'E03.9',  'Hypothyroidism, unspecified',                '2023-02-08'),
    ('P004', 'E11.9',  'Type 2 diabetes mellitus without complications', '2022-04-19'),
    ('P004', 'E78.5',  'Hyperlipidemia, unspecified',                '2022-04-19'),
    ('P006', 'F41.1',  'Generalized anxiety disorder',               '2024-09-02');

INSERT INTO vitals (patient_id, timestamp, bp, hr, temp_f) VALUES
    ('P001', '2024-06-12T10:00', '138/85', 72, 98.4),
    ('P003', '2025-11-04T08:15', '118/74', 64, 98.2),
    ('P004', '2026-01-15T14:00', '146/92', 78, 98.6),
    ('P006', '2025-12-10T11:20', '124/80', 70, 98.5);
