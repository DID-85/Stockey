import pymssql

conn = pymssql.connect(
    server='localhost',
    user='TEST',
    password='TEST',
    database='StockyDB'
)
cursor = conn.cursor()
cursor.execute('SELECT @@VERSION')
row = cursor.fetchone()
print(row)
conn.close()
