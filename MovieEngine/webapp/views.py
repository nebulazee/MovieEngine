from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from . import movie_recommendation_engine

def index(request):
	#return HttpResponse("<h2>HEY! its me Aishwarya</h2>")
	return render(request,'personal/home.html')
def mainPage(request):
	
	#return HttpResponse("<h2>HEY! its me Aishwarya on   page1</h2>")
	return render(request,'personal/mainPage.html')    
@csrf_exempt
def movieRecom(request):
	#print("hi")
	if request.method=='POST':
		data=request.POST['movie']
		#data1=sum.sumfun(data)
		data1=movie_recommendation_engine.recommendMovies(data)		
		context={'data1':data1}
	return render(request,'personal/recommendation.html',context)