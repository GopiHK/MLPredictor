from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

class File(db.Model):
    '''
    Define the following three columns
    1. id - Which stores a file id 
    2. name - Which stores the file name
    3. filepath - Which stores path of file, 
    '''
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), index=True)

    filepath = db.Column(db.String(100), index=True)
    
    def __rep__(self):
        return "<File : {}>".format(self.name)
    


