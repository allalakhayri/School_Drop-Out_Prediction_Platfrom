import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class FormService {
  baseUrl: string ="http://127.0.0.1:5000/";

  constructor(private http: HttpClient) {}

  // add a method to send form data to the Flask backend
  sendForm(formValue: any) {
    console.log(formValue);
    return this.http.post<any>(`${this.baseUrl}predict`, formValue);
  }
}



