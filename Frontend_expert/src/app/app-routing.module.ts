import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {UploadFileComponent} from './file-upload/file-upload.component';
import {DataPreprocessingComponent} from './data-preprocessing/data-preprocessing.component'
import {DatavisualisationComponent} from './datavisualisation/datavisualisation.component'
import {DataProcessingComponent} from './data-processing/data-processing.component'
import { RapportComponent } from './rapport/rapport.component';
import {AlgorithmsComponent} from './algorithms/algorithms.component';
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';
import { HomeComponent } from './home/home.component';
import { UserFormComponent } from './user-form/user-form.component';
const routes: Routes = [
  {path :'' , component : HomeComponent},
  {path :'login' , component : LoginComponent},
  {path :'signup' , component : SignupComponent},
  { path: 'upload', component: UploadFileComponent },
  { path: 'datapreprocessing', component: DataPreprocessingComponent },
  { path: 'datavisualisation', component: DatavisualisationComponent },
  {path: 'dataprocessing', component: DataProcessingComponent},
  {path: 'algorithms', component: AlgorithmsComponent},
  {path: 'report', component: RapportComponent},
  {path: 'user', component: UserFormComponent},

];

@NgModule({ 
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
