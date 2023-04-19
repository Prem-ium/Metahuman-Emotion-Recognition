// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "MyBlueprintFunctionLibrary.generated.h"
#include "MyInterface.h"
#include "Async/AsyncWork.h"
#include "HAL/Runnable.h"

class BLUEPRINTTRAINING_API MyBlueprintFunction : public FRunnable {
public:
	/*MyBlueprintFunction(UObject* object) { this->object = object; }
	UObject* object;

	FORCEINLINE	TStatId GetStatId() const {
		RETURN_QUICK_DECLARE_CYCLE_STAT(MyBlueprintFunction, STATGROUP_ThreadPoolAsyncTasks);
	}

	void DoWork();*/
	MyBlueprintFunction();
	virtual ~MyBlueprintFunction() override;

	bool Init() override;
	uint32 Run() override;
	void Stop() override;


private:
	FRunnableThread* Thread;
	bool bRunThread;
};

/**
 * 
 */

//UCLASS()
//class BLUEPRINTTRAINING_API UMyBlueprintFunctionLibrary : public UBlueprintFunctionLibrary
//{
//	GENERATED_BODY()
//
//
//
//public:
//	UFUNCTION(BlueprintCallable)
//		static void CallMultithreadedFunction(UObject* object) {
//		(new FAutoDeleteAsyncTask<MyBlueprintFunction>(object))->StartBackgroundTask();
//	}
//};