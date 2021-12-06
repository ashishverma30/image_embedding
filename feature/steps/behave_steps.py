from behave import *
@given('Book details')
def impl_bk(context):
   print('Book details entered')
@then('Verify book name')
def impl_bk(context):
   print('Verify book name')