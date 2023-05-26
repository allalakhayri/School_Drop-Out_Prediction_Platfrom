import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatavisualisationComponent } from './datavisualisation.component';

describe('DatavisualisationComponent', () => {
  let component: DatavisualisationComponent;
  let fixture: ComponentFixture<DatavisualisationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DatavisualisationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DatavisualisationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
