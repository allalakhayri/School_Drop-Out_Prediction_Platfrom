import { Component, OnInit } from '@angular/core';
import {FormBuilder,FormControl,FormGroup, Validators} from '@angular/forms'
import {Router} from '@angular/router'
import { UsersService } from '../users.service';
import { ValidateForm } from 'src/Helpers/validateform'
import {NgToastService} from 'ng-angular-popup'
@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
type:string="password";
isText:boolean=false;
eyeIcon:string= "fa-eye-slash";
loginForm!:FormGroup;
constructor(
private fb:FormBuilder,
private route:Router,
private auth:UsersService,
private toast: NgToastService

)
{

}

ngOnInit():void{
this.loginForm=this.fb.group({
email:['',Validators.required],
password:['',Validators.required]
})
}

hideShowPass(){
  this.isText= !this.isText;
  this.isText ? this.eyeIcon="fa-eye": this.eyeIcon="fa-eye-slash"
  this.isText?this.type="text":this.type="password"
}
onLogin(){
  if(this.loginForm.valid){
    //send the obj to database
    this.auth.login(this.loginForm.value).subscribe({
      next:(res=>{
        console.log(res);
        if(res.Message=="Login success !"){
        
        this.auth.storeToken(res.token);
        this.route.navigate(['upload']);
        this.toast.success({detail:"SUCCESS",summary:res.Message,duration:5000});}
         else if(res.Message=="Email does not exist.!")
         this.toast.error({detail:"ERROR",summary:res.Message,duration:5000});
         else          this.toast.error({detail:"ERROR",summary:res.Message,duration:5000});


          }),
      error:(err=>{
       this.toast.error({detail:"ERROR",summary:"Something went wrong !",duration:5000});
      })
    })
  }
  else{
    //throw the error using toaster and with required field
    ValidateForm.vlidateAllFormFields(this.loginForm);
    alert("Your form is invalid !");
  }
}
}