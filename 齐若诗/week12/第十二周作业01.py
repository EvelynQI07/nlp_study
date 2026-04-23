import sqlite3

class SQLQAAgent:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_total_tables(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        return len(tables)
    
    def get_employee_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM employees;")
        result = self.cursor.fetchone()
        return result[0]
    
    def get_customer_and_employee_counts(self):
        self.cursor.execute("SELECT COUNT(*) FROM customers;")
        customer_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM employees;")
        employee_count = self.cursor.fetchone()[0]
        
        return customer_count, employee_count
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db_path = "/Users/evelynq/Desktop/工作相关/大模型LLM/nlp/week12/04_SQL-Code-Agent-Demo/chinook.db"
    agent = SQLQAAgent(db_path)
    
    print("提问1：数据库中总共有多少张表")
    table_count = agent.get_total_tables()
    print(f"回答：数据库中总共有 {table_count} 张表\n")
    
    print("提问2：在原始的数据库中，员工表中有多少条记录")
    employee_count = agent.get_employee_count()
    print(f"回答：员工表中有 {employee_count} 条记录\n")
    
    print("提问3：在数据库中所有客户的个数和员工的个数分别多少？")
    customer_count, emp_count = agent.get_customer_and_employee_counts()
    print(f"回答：客户的个数是 {customer_count}，员工的个数是 {emp_count}")
    
    agent.close()
