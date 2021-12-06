@create_project
Feature: Payment Process
   Background:
      Given launch application
      Then Input credentials
   Scenario: Credit card transaction
      Given user is on credit card payment screen
      Then user should be able to complete credit card payment