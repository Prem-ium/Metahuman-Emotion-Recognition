// Fill out your copyright notice in the Description page of Project Settings.


#include "MyBlueprintFunctionLibrary.h"
#pragma region Main Thread Code

MyBlueprintFunction::MyBlueprintFunction() {
	Thread = FRunnableThread::Create(this, TEXT("MetaHumanThread"));
}

MyBlueprintFunction::~MyBlueprintFunction() {
	if (Thread) {
		Thread->Kill();
		delete Thread;
	}
}

#pragma endregion

bool MyBlueprintFunction::Init() {
	UE_LOG(LogTemp, Warning, TEXT("Custom thread has been initialized"))

		return true;
}

uint32 MyBlueprintFunction::Run() {
	while (bRunThread) {
		UE_LOG(LogTemp, Warning, TEXT("Custom thread is running"))
			FPlatformProcess::Sleep(1.0f);
	}
	return 0;
}

void MyBlueprintFunction::Stop() {
	bRunThread = false;
}
