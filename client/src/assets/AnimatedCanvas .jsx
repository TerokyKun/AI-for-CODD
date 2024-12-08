import React, { useEffect, useRef, useState } from 'react';

const AnimatedCanvas = () => {
  const [animateHeader, setAnimateHeader] = useState(true);
  const largeHeaderRef = useRef(null);
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [target, setTarget] = useState({ x: window.innerWidth / 2, y: window.innerHeight / 2 });

  useEffect(() => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const largeHeader = largeHeaderRef.current;
    largeHeader.style.height = `${height}px`;
    canvas.width = width;
    canvas.height = height;

    // Create points
    const pointsArr = [];
    for (let x = 0; x < width; x += width / 20) {
      for (let y = 0; y < height; y += height / 20) {
        let px = x + Math.random() * (width / 20);
        let py = y + Math.random() * (height / 20);
        pointsArr.push({ x: px, originX: px, y: py, originY: py, angle: Math.random() * Math.PI * 2 });
      }
    }

    // Assign closest points
    pointsArr.forEach((point) => {
      const closest = [];
      pointsArr.forEach((otherPoint) => {
        if (point !== otherPoint) {
          if (closest.length < 5) {
            closest.push(otherPoint);
          } else {
            const maxDist = Math.max(...closest.map((p) => getDistance(point, p)));
            const dist = getDistance(point, otherPoint);
            if (dist < maxDist) {
              closest.push(otherPoint);
              closest.sort((a, b) => getDistance(point, a) - getDistance(point, b));
              closest.pop();
            }
          }
        }
      });
      point.closest = closest;
      point.circle = new Circle(point, 2 + Math.random() * 2, 'rgba(255,255,255,0.3)');
    });

    setPoints(pointsArr);

    // Handle window resize
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      largeHeader.style.height = `${window.innerHeight}px`;
    };

    // Handle scroll
    const handleScroll = () => {
      if (document.body.scrollTop > height) setAnimateHeader(false);
      else setAnimateHeader(true);
    };

    // Handle mouse move
    const handleMouseMove = (e) => {
      const posx = e.pageX || e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
      const posy = e.pageY || e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
      setTarget({ x: posx, y: posy });
    };

    // Attach event listeners
    window.addEventListener('resize', handleResize);
    window.addEventListener('scroll', handleScroll);
    window.addEventListener('mousemove', handleMouseMove);

    // Cleanup event listeners on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  useEffect(() => {
    const animate = () => {
      if (animateHeader) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const targetDistance = target;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        points.forEach((point) => {
          const dist = getDistance(targetDistance, point);
          if (dist < 4000) {
            point.active = 0.3;
            point.circle.active = 0.6;
          } else if (dist < 20000) {
            point.active = 0.1;
            point.circle.active = 0.3;
          } else if (dist < 40000) {
            point.active = 0.02;
            point.circle.active = 0.1;
          } else {
            point.active = 0;
            point.circle.active = 0;
          }

          // Apply smooth shaking effect
          shakePoint(point);

          drawLines(ctx, point);
          point.circle.draw(ctx);
        });
      }
      requestAnimationFrame(animate);
    };

    animate();
  }, [animateHeader, points, target]);

  const drawLines = (ctx, p) => {
    if (!p.active) return;
    p.closest.forEach((closestPoint) => {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(closestPoint.x, closestPoint.y);
      ctx.strokeStyle = `rgba(156,217,249,${p.active})`;
      ctx.stroke();
    });
  };

  const getDistance = (p1, p2) => {
    return Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2);
  };

  const shakePoint = (point) => {
    const shakeRadius = 0.5; // Very small radius for subtle shake
    const shakeSpeed = 0.02; // Slow down the shaking effect

    // Calculate shake based on sine and cosine for smooth, circular motion
    point.x = point.originX + Math.sin(point.angle) * shakeRadius;
    point.y = point.originY + Math.cos(point.angle) * shakeRadius;

    // Increment the angle slowly for more subtle and smooth animation
    point.angle += shakeSpeed;
  };

  class Circle {
    constructor(pos, rad, color) {
      this.pos = pos || null;
      this.radius = rad || null;
      this.color = color || null;
      this.active = 0;
    }

    draw(ctx) {
      if (!this.active) return;
      ctx.beginPath();
      ctx.arc(this.pos.x, this.pos.y, this.radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = `rgba(156,217,249,${this.active})`;
      ctx.fill();
    }
  }

  return (
    <div
      ref={largeHeaderRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'transparent',
        }}
      ></canvas>
    </div>
  );
};

export default AnimatedCanvas;
