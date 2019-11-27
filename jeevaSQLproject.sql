create database BankJ

use  BankJ

CREATE TABLE UserLogins
			(UserLoginID smallint primary key,
			UserLogin char(15),
			UserPassword varchar(20)
			)

CREATE TABLE UserSecurityQuestions
			(UserSecurityQuestionID tinyint primary key,
			UserSecurityQuestion varchar(50)
			)

CREATE TABLE AccountType
			(AccountTypeID tinyint primary key,
			AccountTypeDescription varchar(30)
			
			)

CREATE TABLE SavingsInterestRates
			(InterestSavingsRateID tinyint primary key,
			InterestRateValue numeric(9,9),
			InterestRateDescription varchar(20) 
			
			)

CREATE TABLE AccountStatusType 
			(AccountStatusTypeID tinyint primary key,
			AccountStatusDescription varchar(30)
			
			)

CREATE TABLE Employee
			(EmployeeID int primary key,
			EmployeeFirstName varchar(25),
			EmployeeMiddleInitial char(1),
			EmployeeLastName varchar(25),
			EmployeeIsManager bit
			)

CREATE TABLE TransactionType
			(TransactionTypeID tinyint primary key,
			TransactionTypeName char(10),
			TransactionTypeDescription varchar(50),
			TransactionFeeAmount smallmoney
			
			)

CREATE TABLE LoginErrorLog
			(ErrorLogID int primary key,
			ErrorTime datetime,
			FailedTransactionXML xml
			
			)

CREATE TABLE FailedTransactionErrorType
			(FailedTransactionErrorTypeID tinyint primary key,
			FailedTransactionDescription varchar(50)
			
			)

CREATE TABLE FailedTransactionLog
			(FailedTransactionID int primary key,
			FailedTransactionErrorTypeID tinyint foreign key references FailedTransactionErrorType(FailedTransactionErrorTypeID),
			FailedTransactionErrorTime datetime,
			FailedTransactionXML xml
			
			)

CREATE TABLE UserSecurityAnswers
			(UserLoginID smallint primary key foreign key references UserLogins(UserLoginID),
			UserSecurityAnswer varchar(25),
			UserSecurityQuestionID tinyint foreign key references UserSecurityQuestions
			)


CREATE TABLE Account
			(AccountID int primary key,
			CurrentBalance int,
			AccountTypeID tinyint foreign key references AccountType(AccountTypeID),
			AccountStatusTypeID tinyint foreign key references AccountStatusType(AccountStatusTypeID) ,
			InterestSavingsRateID tinyint foreign key references SavingsInterestRates(InterestSavingsRateID)
			)

CREATE TABLE OverDraftLog
			(AccountID int primary key foreign key references Account(AccountID),
			OverDraftDate datetime,
			OverDraftAmount money,
			OvreDraftTransactionXML xml
			)





CREATE TABLE Login_Account
			(UserLoginID smallint foreign key references UserLogins(UserLoginID),
			AccountID int foreign key references Account(AccountID)
			)


CREATE TABLE Customer
			(CustomerID int primary key,
			AccountID int foreign key references Account(AccountID),
			CustomerAddress1 varchar(30),
			CustomerAddress2 varchar(30),
			CustomerFirstName varchar(30),
			CustomerMiddleInitial char(1),
			CustomerLastName varchar(30),
			City varchar(20),
			Stat char(2),
			ZipCode char(10),
			EmailAddress varchar(40),
			HomePhone char(10),
			CellPhone char(10),
			WorkPhone char(10),
			SSN char(9),
			UserLoginID smallint foreign key references UserLogins(UserLoginID)			
			)


CREATE TABLE Customer_Account
			(AccountID int foreign key references Account(AccountID) ,
			CustomerID int foreign key references Customer(CustomerID)
			
			)

CREATE TABLE TransactionLog 
			(TransactionID int primary key,
			TransactionDate datetime,
			TransactionTypeID tinyint foreign key references TransactionType(TransactionTypeID),
			TransactionAmount money,
			NewBalance money,
			AccountID int foreign key references Account(AccountID),
			CustomerID int foreign key references Customer(CustomerID),
			EmployeeID int foreign key references Employee(EmployeeID),
			UserLoginID smallint foreign key references UserLogins(UserLoginID)
			)
INSERT INTO UserLogins (UserLoginID, UserLogin, UserPassword)
Values	(1, 'jeevamaryloui3','Aa123456'),
		(2, 'vinuvsunny3','Bb123456'),
		(3, 'antonyloui3','Cc123456'),
		(4, 'sonuvsunny3','Dd123456'),
		(5, 'steffytresa3','Ee123456');

INSERT INTO UserSecurityQuestions (UserSecurityQuestionID,UserSecurityQuestion) 
values	(1,'What is your favourite colour'),
		(2,'What is your favourite sport'),
		(3,'What is your favourite book');

INSERT INTO AccountType (AccountTypeID,AccountTypeDescription)
values	(1,'ckecking'),
		(2,'savings');

INSERT INTO SavingsInterestRates (InterestSavingsRateID,InterestRateValue,InterestRateDescription)
values	(1,0.050,'Enhanced saving'),
		(2,0.020,'Day to day saving'),
		(3,0.010,'Young saving');

INSERT INTO AccountStatusType (AccountStatusTypeID, AccountStatusDescription)
values	(1,'Active'),
		(2,'Pending approval'),
		(3,'Not yet activated'),
		(4,'Inactive');

INSERT INTO Employee (EmployeeID,EmployeeFirstName,EmployeeMiddleInitial,EmployeeLastName,EmployeeIsManager)
values	(1,'John','F','Epsic',1),
		(2,'Reem','H','David',0),
		(3,'Peter','I','Smith',1);

INSERT INTO TransactionType (TransactionTypeID, TransactionTypeName, TransactionTypeDescription,TransactionFeeAmount) 
values	(1,'Deposite','deposite cash into your account',1),
		(2,'Withdraw','withdraw cash from your account',2);

INSERT INTO LoginErrorLog (ErrorLogID,ErrorTime) 
values	(1,GETDATE()),
		(2,GETDATE());

INSERT INTO FailedTransactionErrorType (FailedTransactionErrorTypeID,FailedTransactionDescription) 
values	(1,'Exceeded withdrawl amount limit '),
		(2,'Have not sufficient balance ');

INSERT INTO UserSecurityAnswers (UserLoginID,UserSecurityAnswer,UserSecurityQuestionID) 
values  (1,'green',1),
		(2,'football',2),
		(3,'machine learning',3);

INSERT INTO FailedTransactionLog (FailedTransactionID,FailedTransactionErrorTypeID,FailedTransactionErrorTime) 
values	(1,1,GETDATE()),
		(2,1,GETDATE());

INSERT INTO Account (AccountID,CurrentBalance,AccountTypeID,AccountStatusTypeID,InterestSavingsRateID) 
values	(10001,8500,1,1,1),
		(10002,500,2,1,1),
		(10003,1500,1,2,1),
		(10004,800,2,1,2);

INSERT INTO OverDraftLog (AccountID,OverDraftDate,OverDraftAmount) 
values	(10001,GETDATE(),250),
		(10003,GETDATE(),300);

INSERT INTO Login_Account (UserLoginID,AccountID) 
values	(1,10001),
		(2,10003);

INSERT INTO Customer (CustomerID,AccountID,CustomerAddress1,CustomerAddress2,CustomerFirstName,
CustomerMiddleInitial,CustomerLastName,City,Stat,ZipCode,EmailAddress,HomePhone,CellPhone,WorkPhone,SSN,UserLoginID)
values	(1,10001,'2788 KEELE','1514','jeeva','F','Loui','North York','ON','M3M2G2','louijeeva@gamil','4168767490','6475392510','6136086267','1525',1);

 INSERT INTO Customer_Account (AccountID,CustomerID) 
values	(10001,1),
		(10002,1);

INSERT INTO TransactionLog (TransactionID,TransactionDate,TransactionTypeID,TransactionAmount,NewBalance,AccountID,CustomerID,EmployeeID,UserLoginID)
values	(1,GETDATE(),1,550,4500,10001,1,1,1);

/* 1. Create a view to get all customers with checking account from ON province. */
CREATE VIEW ON_customers1
as select cu.AccountID,cu.CustomerLastName,cu.Stat ,cu.CustomerFirstName,ac.AccountTypeID
from Customer as CU, Account as AC

where CU.AccountID= AC.AccountID and CU.stat = 'ON'

select * from ON_customers1
select * from Customer_Account

/* 2. Create a view to get all customers with total account balance (including interest rate)*/

 create view CusTotalACC

 as
 
 select CONCAT(c.CustomerFirstName,'',c.CustomerLastName) as fullname, a.CurrentBalance*(1+s.InterestRateValue) as TotalAccouny_balance
 from Customer C, Account A, SavingsInterestRates S, Customer_Account CA

 where c.CustomerID=ca.CustomerID and ca.AccountID=a.AccountID and a.InterestSavingsRateID=s.InterestSavingsRateID 
 and a.CurrentBalance*(1+s.InterestRateValue)<=5000

 select * from CusTotalACC
/* 3. Create a view to get counts of checking and savings accounts by customer.*/
create view co1
as
select CU.CustomerFirstName, count(A.AccountTypeID) as cccc
from Customer as CU,  Account as A,Customer_Account as CA
where Cu.AccountID = A.AccountID
group by cu.CustomerFirstName

select * from co1


/* 4. Create a view to get any particular user’s login and password using AccountId. */

create view user_login5
as
select ul.UserLogin, Ul.UserPassword
from Login_Account LA, UserLogins UL

where la.UserLoginID=ul.UserLoginID  and la.AccountID=10001

select * from user_login5

/* 5. Create a view to get all customers’ overdraft amount. */

create view overdraft_amount
as 

select c.AccountID, c.CustomerFirstName, od.OverDraftAmount
from customer C,Account A, OverDraftLog OD
where c.AccountID=a.AccountID

select * from overdraft_amount

/* 6. Create a stored procedure to add “User_” as a prefix to everyone’s login (username). */

create view userlogins_copy
as select * from UserLogins

select * from userlogins_copy

create proc user_prefix1
as
select concat('USER_', UserLogin) as User_prfix from userlogins_copy

exec user_prefix1

/* 7. Create a stored procedure that accepts AccountId as a parameter and returns customer’s full name */

create proc fullname1 @accountID int
as

select concat(CustomerFirstName,CustomerMiddleInitial,CustomerLastName) as fullname
from Customer
where AccountID=@accountID

exec fullname1 10001

/* 8. Create a stored procedure that returns error logs inserted in the last 24 hours*/

create proc error_log6
as

select  ErrorTime,datediff(day,ErrorTime,GETDATE()) as recenterror from LoginErrorLog
where datediff(day,ErrorTime,GETDATE())<=1

exec error_log6

/*9.	Create a stored procedure that takes a deposit as a parameter and updates CurrentBalance value for that particular account*/
create proc depo12 @trans_amount money

as

select a.AccountTypeID,a.CurrentBalance+ @trans_amount as CurrentBalance

from Account as a right join TransactionLog as t
on t.AccountID = a.AccountID 
where t.TransactionTypeID=1  
exec depo12 200

 

/*10.	Create a stored procedure that takes a withdrawal amount as a parameter and updates CurrentBalance value for that particular account*/

create proc withdraw3 @trans_amount money

as

select a.AccountTypeID,a.CurrentBalance- @trans_amount as CurrentBalance

from Account as a right join TransactionLog as t
on t.AccountID = a.AccountID 
where t.TransactionTypeID=2 
exec withdraw3 100




/*12.	Delete all error logs created in the last hour*/
delete from LoginErrorLog 
where datediff(HOUR, ErrorTime, getdate())<=1

/*13.	Write a query to remove SSN column from Customer table*/

alter table customer drop column SSN;
