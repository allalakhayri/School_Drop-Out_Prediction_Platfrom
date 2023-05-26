import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http'; // Import HttpClientModule
import { FormsModule, ReactiveFormsModule } from '@angular/forms'; // Import ReactiveFormsModule
import { AppComponent } from './app.component';
import { UploadFileComponent } from './file-upload/file-upload.component';
import { RouterModule } from '@angular/router';
import { DataPreprocessingComponent } from './data-preprocessing/data-preprocessing.component';
import { AppRoutingModule } from './app-routing.module';
import { NgChartsModule } from 'ng2-charts';
import { DatavisualisationComponent } from './datavisualisation/datavisualisation.component';
import { DataProcessingComponent } from './data-processing/data-processing.component';
import { RapportComponent } from './rapport/rapport.component';
import { AlgorithmsComponent } from './algorithms/algorithms.component';
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';
import { HomeComponent } from './home/home.component';
import { NgToastComponent, NgToastModule } from 'ng-angular-popup';
import { UserFormComponent } from './user-form/user-form.component';

@NgModule({
  declarations: [
    AppComponent,
    UploadFileComponent,
    DataPreprocessingComponent,
    DatavisualisationComponent,
    DataProcessingComponent,
    AlgorithmsComponent,
    RapportComponent,
    LoginComponent,
    SignupComponent,
    HomeComponent,
    UserFormComponent,
      ],
  imports: [
    BrowserModule,
    HttpClientModule,
    ReactiveFormsModule ,
    FormsModule, 
    RouterModule,
    AppRoutingModule ,
    NgChartsModule,
    NgToastModule,
    
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
