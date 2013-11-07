#include <iostream>
#include "gtest/gtest.h"
using namespace std;

//int main(){
//cout<<"hello"<<endl;
//}

TEST(helloTest, testTest)
{
   for(int i=0;i<100;i++){
           EXPECT_EQ(1,1);
    }
}
