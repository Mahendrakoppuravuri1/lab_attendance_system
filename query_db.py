import sqlite3

# Connect to the database
conn = sqlite3.connect('C:\\Users\\mahen\\Documents\\new_VCRC\\attendance.db')
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print("Tables:", tables)

# Query the attendance table
c.execute("SELECT * FROM attendance;")
rows = c.fetchall()

print("Attendance Records:")
for row in rows:
    print(row)

conn.close()
