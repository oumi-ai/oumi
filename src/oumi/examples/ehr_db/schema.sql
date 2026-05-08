-- EHR demo schema (SQLite-compatible; uses standard SQL where possible).

CREATE TABLE patients (
    patient_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    dob        TEXT NOT NULL,
    status     TEXT NOT NULL
);

CREATE TABLE allergies (
    patient_id TEXT NOT NULL,
    substance  TEXT NOT NULL,
    PRIMARY KEY (patient_id, substance),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE medications (
    patient_id TEXT NOT NULL,
    name       TEXT NOT NULL,
    dose       TEXT NOT NULL,
    PRIMARY KEY (patient_id, name),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE diagnoses (
    patient_id  TEXT NOT NULL,
    code        TEXT NOT NULL,
    description TEXT NOT NULL,
    date        TEXT NOT NULL,
    PRIMARY KEY (patient_id, code),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE vitals (
    patient_id TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    bp         TEXT NOT NULL,
    hr         INTEGER NOT NULL,
    temp_f     REAL NOT NULL,
    PRIMARY KEY (patient_id, timestamp),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
