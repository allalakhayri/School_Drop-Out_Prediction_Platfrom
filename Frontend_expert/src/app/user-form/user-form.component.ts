import { Component, OnInit } from '@angular/core';
import {FormService} from './form.service'
import { HttpHeaders } from '@angular/common/http';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import {Chart} from 'chart.js/auto'
@Component({
  selector: 'app-user-form',
  templateUrl: './user-form.component.html',
  styleUrls: ['./user-form.component.css']
})
export class UserFormComponent implements OnInit {
  form!:FormGroup;
  title: any;
  myChar!:Chart
  constructor(private fb:FormBuilder,
    private route:Router,private formService: FormService) {

    }
ngOnInit():void{
  this.form=this.fb.group({
    Gender:['',Validators.required],
    marital_status:['',Validators.required],
    age: ['',Validators.required],
    scolarship: ['',Validators.required],
    mother_qualification: ['',Validators.required],
    mother_occupation: ['',Validators.required],
    father_occupation: ['',Validators.required],
    inflation_rate: ['',Validators.required],
    previous_qualifcation: ['',Validators.required],
    course: ['',Validators.required],
    curricular_units_1st_sem_without_evaluations: ['',Validators.required],
    curricular_units_1st_sem_approved: ['',Validators.required],
    curricular_units_1st_sem_credited: ['',Validators.required],
    curricular_units_1st_sem_with_evaluations: ['',Validators.required],
    curricular_units_2nd_sem_without_evaluations: ['',Validators.required],
    curricular_units_2nd_sem_approved: ['',Validators.required],
    curricular_units_2nd_sem_credited: ['',Validators.required],
    curricular_units_2nd_sem_with_evaluations: ['',Validators.required]
    })
     this.myChar=new Chart("myChart", {
      type: 'bar',
      data: {
        labels: ['Graduate', 'Dropout', 'Enrolled'],
        datasets: [{
          label: 'Decision',
          data: [0, 0,0],
          borderWidth: 1
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });

}
  onSubmit() {
   console.log(this.form.value);
    // send the form data to the Flask backend using the FormService
    this.formService.sendForm(this.form.value).subscribe(
    {next:(
      (      response: { prediction: string; })=>{
        console.log(response.prediction);
        if(response.prediction=="Graduate"){

          this.myChar.data.datasets[0].data[0] = 100;
          this.myChar.update()
          this.form.reset();

        }
        else    if(response.prediction=="Dropout"){
          this.myChar.data.datasets[0].data[0] = 0;
          this.myChar.data.datasets[0].data[1] = 100;
          this.myChar.update()
          this.form.reset();

        }
        else   {
          this.myChar.data.datasets[0].data[1] = 0;
          this.myChar.data.datasets[0].data[2] = 100;
          this.myChar.update()
          this.form.reset();

        }

      }
    ),error:((err: any)=>{

      console.log(err);
     } )
    })
  }

}
