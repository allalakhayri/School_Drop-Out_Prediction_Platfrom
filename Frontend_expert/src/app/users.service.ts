import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {map} from 'rxjs/operators';
import { Users } from '../user.model'
import { Router } from '@angular/router';
@Injectable({
  providedIn: 'root'
})
export class UsersService {
  baseUrl: string ="http://127.0.0.1:5000/";

  constructor(private httpClient: HttpClient,private route:Router) { }

  public signUp(userObj:any){
    return this.httpClient.post<any>(`${this.baseUrl}Users`,userObj);
  }
  public login(loginObj:any){
    return this.httpClient.post<any>(`${this.baseUrl}UsersLogin`,loginObj);
  }
  storeToken(tokenValue:string){
    localStorage.setItem('token',tokenValue);
  }
  getToken(){
    return localStorage.getItem('token');
  }
  isLoggedIn():boolean{
    return !!localStorage.getItem('token');
  }

  signOut(){
    localStorage.clear();
    this.route.navigate(['login'])
  }
}