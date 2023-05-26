import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';


@Injectable({
  providedIn: 'root'
})
export class GenerateService {

private apiUrl = 'http://localhost:5000/report'; 

constructor(private http: HttpClient) { }

generateData(data :any ) {
  return this.http.post(this.apiUrl, data);
}
}

