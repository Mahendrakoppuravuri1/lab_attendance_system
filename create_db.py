import sqlite3

# Create a new SQLite database and connection
conn = sqlite3.connect('attendance.db')
c = conn.cursor()

# Create table for attendance records
c.execute('''
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    registration_number TEXT NOT NULL,
    time_in TEXT,
    time_out TEXT
)
''')

print("Database and table created successfully!")

conn.commit()
conn.close()
